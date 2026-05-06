#!/usr/bin/env python
"""Train a PyTorch fusion-vs-receiver correctness detector with epoch logs.

This mirrors train_fusion_receiver_correctness_detector.py:
- default action is fusion
- positive class means "receiver beats fusion"
- BCE is trained only on differential samples where one source is correct
- validation accuracy/net are measured on the full validation split
"""

import argparse
import copy
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_fusion_receiver_correctness_detector import (
    feature_names,
    load_rows,
    _is_correct,
)


class TorchCorrectnessDetector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        architecture: str,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if architecture == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif architecture == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def rapid_warmup_sine_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    warmup_start_multiplier: float,
    min_multiplier: float,
) -> float:
    if total_steps <= 1:
        return 1.0
    if step <= warmup_steps:
        progress = step / max(1, warmup_steps)
        return warmup_start_multiplier + (1.0 - warmup_start_multiplier) * progress

    decay_steps = max(1, total_steps - warmup_steps)
    decay_progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    sine_decay = np.sin((1.0 - decay_progress) * np.pi / 2.0)
    return float(min_multiplier + (1.0 - min_multiplier) * sine_decay)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def row_key(row: Dict[str, str]) -> Tuple[str, str]:
    return str(row.get("subject", "")), str(row.get("question_id", ""))


def load_label_rows(label_csv: Path, feature_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    with label_csv.open(newline="") as f:
        label_map = {row_key(row): row for row in csv.DictReader(f)}

    label_rows: List[Dict[str, str]] = []
    missing: List[Tuple[str, str]] = []
    for row in feature_rows:
        key = row_key(row)
        label_row = label_map.get(key)
        if label_row is None:
            missing.append(key)
        else:
            label_rows.append(label_row)

    if missing:
        preview = ", ".join(f"{subject}#{question_id}" for subject, question_id in missing[:10])
        raise ValueError(
            f"label_csv is missing {len(missing)} rows from input_csv; first missing: {preview}"
        )
    return label_rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    X: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=device)
        prob = torch.sigmoid(model(batch)).detach().cpu().numpy()
        probs.append(prob)
    if not probs:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(probs, axis=0).astype(np.float64)


@torch.no_grad()
def compute_loss(
    model: nn.Module,
    X: np.ndarray,
    utility: np.ndarray,
    indices: np.ndarray,
    criterion: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    diff_indices = indices[utility[indices] != 0]
    if len(diff_indices) == 0:
        return 0.0
    losses = []
    weights = []
    for start in range(0, len(diff_indices), batch_size):
        batch_idx = diff_indices[start:start + batch_size]
        batch_x = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=device)
        batch_y = torch.as_tensor((utility[batch_idx] > 0).astype(np.float32), device=device)
        loss = criterion(model(batch_x), batch_y)
        losses.append(float(loss.detach().cpu()) * len(batch_idx))
        weights.append(len(batch_idx))
    return float(np.sum(losses) / max(1, np.sum(weights)))


def evaluate_threshold_arrays(
    receiver_ok: np.ndarray,
    fusion_ok: np.ndarray,
    utility: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    choose_receiver = probabilities >= threshold
    correct = np.where(choose_receiver, receiver_ok, fusion_ok)
    help_count = np.logical_and(choose_receiver, utility > 0).sum()
    harm_count = np.logical_and(choose_receiver, utility < 0).sum()
    total = int(len(probabilities))
    receiver_count = int(choose_receiver.sum())
    return {
        "threshold": float(threshold),
        "accuracy": float(correct.sum() / total) if total else 0.0,
        "correct": int(correct.sum()),
        "total": total,
        "help": int(help_count),
        "harm": int(harm_count),
        "net": int(help_count - harm_count),
        "receiver_count": receiver_count,
        "selected_source_counts": {
            "receiver": receiver_count,
            "fusion": int(total - receiver_count),
        },
    }


def tune_threshold_arrays(
    receiver_ok: np.ndarray,
    fusion_ok: np.ndarray,
    utility: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_threshold: float,
    threshold_steps: int,
) -> Dict[str, object]:
    best = None
    for threshold in np.linspace(min_threshold, 1.0, threshold_steps):
        metrics = evaluate_threshold_arrays(receiver_ok, fusion_ok, utility, probabilities, float(threshold))
        key = (
            metrics["correct"],
            metrics["net"],
            -metrics["harm"],
            -metrics["receiver_count"],
        )
        if best is None or key > best[0]:
            best = (key, metrics)
    return best[1]


def evaluate_split(
    model: nn.Module,
    X: np.ndarray,
    utility: np.ndarray,
    receiver_ok: np.ndarray,
    fusion_ok: np.ndarray,
    indices: np.ndarray,
    criterion: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    min_threshold: float,
    fixed_threshold: float,
    threshold_steps: int,
) -> Dict[str, object]:
    probabilities = predict_probabilities(
        model,
        X,
        indices,
        device=device,
        batch_size=batch_size,
    )
    split_utility = utility[indices]
    split_receiver_ok = receiver_ok[indices]
    split_fusion_ok = fusion_ok[indices]
    diff_mask = split_utility != 0
    if diff_mask.any() and np.unique(split_utility[diff_mask]).size == 2:
        auc = float(roc_auc_score((split_utility[diff_mask] > 0).astype(np.int64), probabilities[diff_mask]))
    else:
        auc = 0.0
    best = tune_threshold_arrays(
        split_receiver_ok,
        split_fusion_ok,
        split_utility,
        probabilities,
        min_threshold=min_threshold,
        threshold_steps=threshold_steps,
    )
    fixed = evaluate_threshold_arrays(
        split_receiver_ok,
        split_fusion_ok,
        split_utility,
        probabilities,
        fixed_threshold,
    )
    return {
        "loss": compute_loss(model, X, utility, indices, criterion, device=device, batch_size=batch_size),
        "auc": auc,
        "best": best,
        "fixed": fixed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV used only for labels. Features still come from input-csv. "
            "Use this to train on generated receiver/fusion correctness while "
            "reusing logits-probability features."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-set", choices=["basic38", "logpair_v1", "compact_v1"], default="compact_v1")
    parser.add_argument("--architecture", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "rapid_warmup_sine"], default="constant")
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--warmup-start-lr-multiplier", type=float, default=0.05)
    parser.add_argument("--min-lr-multiplier", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--positive-class-weight", type=float, default=4.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode",
        choices=["random", "group"],
        default="random",
        help="Validation split mode. group holds out whole subjects.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--fixed-threshold", type=float, default=0.796)
    parser.add_argument("--threshold-steps", type=int, default=1001)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_net", "val_accuracy", "val_auc", "val_loss", "val_fixed_net"],
        default="val_net",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    rows, X_raw, utility, train_mask, _y, groups = load_rows(args.input_csv, args.feature_set)
    label_rows = load_label_rows(args.label_csv, rows) if args.label_csv is not None else rows
    receiver_ok = np.asarray([_is_correct(row, "dual_receiver_pred") for row in label_rows], dtype=bool)
    fusion_ok = np.asarray([_is_correct(row, "dual_fusion_pred") for row in label_rows], dtype=bool)
    utility = receiver_ok.astype(np.int64) - fusion_ok.astype(np.int64)
    all_indices = np.arange(len(rows))
    if args.split_mode == "group":
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.val_fraction,
            random_state=args.seed,
        )
        train_indices, val_indices = next(splitter.split(all_indices, groups=groups))
    else:
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=args.val_fraction,
            random_state=args.seed,
            stratify=utility,
        )
    train_diff_indices = train_indices[utility[train_indices] != 0]
    y_train = (utility[train_diff_indices] > 0).astype(np.float32)
    train_subjects = sorted(set(groups[train_indices].tolist()))
    val_subjects = sorted(set(groups[val_indices].tolist()))

    feature_mean = X_raw[train_indices].mean(axis=0, keepdims=True)
    feature_std = X_raw[train_indices].std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    X = ((X_raw - feature_mean) / feature_std).astype(np.float32)

    device = torch.device(args.device)
    model = TorchCorrectnessDetector(
        input_dim=X.shape[1],
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(args.positive_class_weight), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TensorDataset(
        torch.as_tensor(X[train_diff_indices], dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    total_train_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(round(total_train_steps * args.warmup_fraction))
    warmup_steps = min(max(1, warmup_steps), total_train_steps)
    scheduler = None
    if args.lr_schedule == "rapid_warmup_sine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: rapid_warmup_sine_lr_multiplier(
                step,
                total_steps=total_train_steps,
                warmup_steps=warmup_steps,
                warmup_start_multiplier=args.warmup_start_lr_multiplier,
                min_multiplier=args.min_lr_multiplier,
            ),
        )

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "input_csv": str(args.input_csv),
                "label_csv": str(args.label_csv) if args.label_csv is not None else None,
                "output_dir": str(args.output_dir),
                "num_examples": len(rows),
                "num_train_examples": int(len(train_indices)),
                "num_val_examples": int(len(val_indices)),
                "num_train_subjects": int(len(train_subjects)),
                "num_val_subjects": int(len(val_subjects)),
                "num_train_differential_examples": int(len(train_diff_indices)),
                "feature_dim": int(X.shape[1]),
                "total_train_steps": int(total_train_steps),
                "warmup_steps": int(warmup_steps),
            },
        )

    best_key: Optional[Tuple[int, float, float, float]] = None
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += float(loss.detach().cpu()) * batch_x.shape[0]
            total_seen += batch_x.shape[0]

        train_loss = total_loss / max(1, total_seen)
        lr_now = current_lr(optimizer)
        val_metrics = evaluate_split(
            model,
            X,
            utility,
            receiver_ok,
            fusion_ok,
            val_indices,
            criterion,
            device=device,
            batch_size=args.batch_size,
            min_threshold=args.min_threshold,
            fixed_threshold=args.fixed_threshold,
            threshold_steps=args.threshold_steps,
        )
        train_metrics = evaluate_split(
            model,
            X,
            utility,
            receiver_ok,
            fusion_ok,
            train_indices,
            criterion,
            device=device,
            batch_size=args.batch_size,
            min_threshold=args.min_threshold,
            fixed_threshold=args.fixed_threshold,
            threshold_steps=args.threshold_steps,
        )
        val_best = val_metrics["best"]
        if args.checkpoint_metric == "val_net":
            key = (
                int(val_best["net"]),
                float(val_best["accuracy"]),
                float(val_metrics["auc"]),
                -float(val_metrics["loss"]),
            )
        elif args.checkpoint_metric == "val_accuracy":
            key = (
                float(val_best["accuracy"]),
                int(val_best["net"]),
                float(val_metrics["auc"]),
                -float(val_metrics["loss"]),
            )
        elif args.checkpoint_metric == "val_auc":
            key = (
                float(val_metrics["auc"]),
                -float(val_metrics["loss"]),
                int(val_best["net"]),
                float(val_best["accuracy"]),
            )
        elif args.checkpoint_metric == "val_loss":
            key = (
                -float(val_metrics["loss"]),
                float(val_metrics["auc"]),
                int(val_best["net"]),
                float(val_best["accuracy"]),
            )
        elif args.checkpoint_metric == "val_fixed_net":
            key = (
                int(val_metrics["fixed"]["net"]),
                float(val_metrics["fixed"]["accuracy"]),
                float(val_metrics["auc"]),
                -float(val_metrics["loss"]),
            )
        else:
            raise ValueError(f"Unknown checkpoint_metric: {args.checkpoint_metric}")
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val = copy.deepcopy(val_metrics)

        epoch_summary = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_auc": train_metrics["auc"],
            "train_best": train_metrics["best"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_best": val_best,
            "val_fixed": val_metrics["fixed"],
        }
        history.append(epoch_summary)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/lr": lr_now,
                    "train/loss": train_loss,
                    "train/auc": train_metrics["auc"],
                    "train/net": train_metrics["best"]["net"],
                    "train/accuracy": train_metrics["best"]["accuracy"],
                    "train/threshold": train_metrics["best"]["threshold"],
                    "val/loss": val_metrics["loss"],
                    "val/auc": val_metrics["auc"],
                    "val/net": val_best["net"],
                    "val/accuracy": val_best["accuracy"],
                    "val/threshold": val_best["threshold"],
                    "val/help": val_best["help"],
                    "val/harm": val_best["harm"],
                    "val/receiver_count": val_best["receiver_count"],
                    "val/receiver_rate": val_best["receiver_count"] / max(1, val_best["total"]),
                    "val_fixed/net": val_metrics["fixed"]["net"],
                    "val_fixed/accuracy": val_metrics["fixed"]["accuracy"],
                    "val_fixed/help": val_metrics["fixed"]["help"],
                    "val_fixed/harm": val_metrics["fixed"]["harm"],
                },
                step=epoch,
            )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"lr={lr_now:.3e} "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_best['accuracy'] * 100:.2f}% "
            f"val_net={val_best['net']:+d} val_threshold={val_best['threshold']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = evaluate_split(
        model,
        X,
        utility,
        receiver_ok,
        fusion_ok,
        train_indices,
        criterion,
        device=device,
        batch_size=args.batch_size,
        min_threshold=args.min_threshold,
        fixed_threshold=args.fixed_threshold,
        threshold_steps=args.threshold_steps,
    )
    final_val = evaluate_split(
        model,
        X,
        utility,
        receiver_ok,
        fusion_ok,
        val_indices,
        criterion,
        device=device,
        batch_size=args.batch_size,
        min_threshold=args.min_threshold,
        fixed_threshold=args.fixed_threshold,
        threshold_steps=args.threshold_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.output_dir / f"fusion_receiver_correctness_detector_torch_{args.architecture}_{timestamp}.pt"
    summary_path = args.output_dir / f"fusion_receiver_correctness_detector_torch_{args.architecture}_{timestamp}_summary.json"
    config_path = args.output_dir / f"fusion_receiver_correctness_detector_torch_{args.architecture}_{timestamp}_config.json"

    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": int(X.shape[1]),
        "architecture": args.architecture,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "feature_mean": torch.as_tensor(feature_mean.astype(np.float32)),
        "feature_std": torch.as_tensor(feature_std.astype(np.float32)),
    }
    torch.save(checkpoint, model_path)

    summary = {
        "input_csv": str(args.input_csv),
        "label_csv": str(args.label_csv) if args.label_csv is not None else None,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "feature_set": args.feature_set,
        "architecture": args.architecture,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "warmup_fraction": args.warmup_fraction,
        "warmup_start_lr_multiplier": args.warmup_start_lr_multiplier,
        "min_lr_multiplier": args.min_lr_multiplier,
        "checkpoint_metric": args.checkpoint_metric,
        "total_train_steps": int(total_train_steps),
        "warmup_steps": int(warmup_steps),
        "weight_decay": args.weight_decay,
        "positive_class_weight": args.positive_class_weight,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "num_examples": len(rows),
        "num_train_examples": int(len(train_indices)),
        "num_val_examples": int(len(val_indices)),
        "num_train_subjects": int(len(train_subjects)),
        "num_val_subjects": int(len(val_subjects)),
        "val_subjects": val_subjects,
        "num_train_differential_examples": int(len(train_diff_indices)),
        "num_val_differential_examples": int((utility[val_indices] != 0).sum()),
        "feature_names": feature_names(args.feature_set, num_options=4),
        "best_epoch": int(best_epoch),
        "best_val": best_val,
        "final_train": final_train,
        "final_val": final_val,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config = {
        "detector_type": "torch_binary",
        "model_path": str(model_path),
        "threshold": float(best_val["best"]["threshold"]),
        "feature_source": "dual_receiver_probs + dual_fusion_probs",
        "feature_set": args.feature_set,
        "default_source": "fusion",
        "positive_action": "receiver",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if wandb_run is not None:
        wandb_run.summary["model_path"] = str(model_path)
        wandb_run.summary["summary_path"] = str(summary_path)
        wandb_run.summary["config_path"] = str(config_path)
        wandb_run.summary["best_epoch"] = int(best_epoch)
        wandb_run.summary["best_val_net"] = int(best_val["best"]["net"])
        wandb_run.summary["best_val_accuracy"] = float(best_val["best"]["accuracy"])
        wandb_run.summary["best_val_auc"] = float(best_val["auc"])
        wandb_run.summary["best_threshold"] = float(best_val["best"]["threshold"])
        wandb_run.summary["checkpoint_metric"] = args.checkpoint_metric
        wandb_run.finish()

    print(f"Saved model to {model_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved config to {config_path}")
    print(
        f"best_epoch={best_epoch} "
        f"val_auc={best_val['auc']:.4f} "
        f"val_acc={best_val['best']['accuracy'] * 100:.2f}% "
        f"val_net={best_val['best']['net']:+d} "
        f"threshold={best_val['best']['threshold']:.3f}"
    )


if __name__ == "__main__":
    main()
