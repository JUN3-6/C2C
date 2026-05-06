#!/usr/bin/env python
"""Train a cheap fusion-vs-receiver correctness detector from cached dual runs.

The detector is intentionally asymmetric:
- default action is fusion
- switch to receiver only when the detector predicts receiver will beat fusion

Training labels use only differential samples:
- y=1: receiver correct and fusion wrong
- y=0: fusion correct and receiver wrong

Both-correct and both-wrong samples are kept for threshold evaluation but are not
used to fit the decision boundary, because the source choice has zero utility.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _is_correct(row: Dict[str, str], pred_key: str) -> bool:
    pred = str(row.get(pred_key, "") or "").strip().upper()
    true = str(row.get("true_answer", "") or "").strip().upper()
    return bool(pred) and pred == true


def _entropy(probs: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def _top1_margin(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    if probs.size < 2:
        return 0.0
    top2 = np.partition(probs, -2)[-2:]
    return float(top2.max() - top2.min())


def _one_hot(idx: int, size: int) -> List[float]:
    out = [0.0] * size
    if 0 <= idx < size:
        out[idx] = 1.0
    return out


def _safe_log(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _basic_feature_names(num_options: int = 4) -> List[str]:
    letters = [chr(65 + i) for i in range(num_options)]
    names: List[str] = []
    names.extend([f"receiver_prob_{letter}" for letter in letters])
    names.extend([f"fusion_prob_{letter}" for letter in letters])
    names.extend([f"prob_diff_receiver_minus_fusion_{letter}" for letter in letters])
    names.extend([f"prob_ratio_receiver_over_fusion_{letter}" for letter in letters])
    names.extend([f"receiver_top1_is_{letter}" for letter in letters])
    names.extend([f"fusion_top1_is_{letter}" for letter in letters])
    names.extend([
        "receiver_top1_score",
        "fusion_top1_score",
        "top1_score_diff_receiver_minus_fusion",
        "receiver_top1_margin",
        "fusion_top1_margin",
        "top1_margin_diff_receiver_minus_fusion",
        "receiver_entropy",
        "fusion_entropy",
        "entropy_diff_fusion_minus_receiver",
        "same_top1",
        "receiver_prob_on_fusion_top1",
        "fusion_prob_on_receiver_top1",
        "receiver_top1_minus_receiver_prob_on_fusion_top1",
        "fusion_top1_minus_fusion_prob_on_receiver_top1",
        "receiver_sorted_top1_margin",
        "fusion_sorted_top1_margin",
    ])
    return names


def _logpair_feature_names(num_options: int = 4) -> List[str]:
    letters = [chr(65 + i) for i in range(num_options)]
    names: List[str] = []
    names.extend([f"receiver_prob_{letter}" for letter in letters])
    names.extend([f"fusion_prob_{letter}" for letter in letters])
    names.extend([f"prob_diff_receiver_minus_fusion_{letter}" for letter in letters])
    names.extend([f"abs_prob_diff_{letter}" for letter in letters])
    names.extend([f"receiver_log_prob_{letter}" for letter in letters])
    names.extend([f"fusion_log_prob_{letter}" for letter in letters])
    names.extend([f"log_prob_diff_receiver_minus_fusion_{letter}" for letter in letters])
    names.extend([f"abs_log_prob_diff_{letter}" for letter in letters])
    names.extend([f"receiver_x_fusion_prob_{r}{f}" for r in letters for f in letters])
    names.extend([f"receiver_top1_is_{letter}" for letter in letters])
    names.extend([f"fusion_top1_is_{letter}" for letter in letters])
    names.extend([f"top_pair_is_{r}{f}" for r in letters for f in letters])
    names.extend([
        "receiver_top1_score",
        "fusion_top1_score",
        "top1_score_diff_receiver_minus_fusion",
        "receiver_top1_margin",
        "fusion_top1_margin",
        "top1_margin_diff_receiver_minus_fusion",
        "receiver_entropy",
        "fusion_entropy",
        "entropy_diff_fusion_minus_receiver",
        "same_top1",
        "receiver_prob_on_fusion_top1",
        "fusion_prob_on_receiver_top1",
        "receiver_top1_minus_receiver_prob_on_fusion_top1",
        "fusion_top1_minus_fusion_prob_on_receiver_top1",
        "kl_receiver_to_fusion",
        "kl_fusion_to_receiver",
        "js_receiver_fusion",
        "l1_receiver_fusion",
        "l2_receiver_fusion",
        "dot_receiver_fusion",
        "cosine_receiver_fusion",
        "receiver_sorted_top1_margin",
        "fusion_sorted_top1_margin",
    ])
    return names


def _compact_feature_names(num_options: int = 4) -> List[str]:
    letters = [chr(65 + i) for i in range(num_options)]
    names: List[str] = []
    names.extend([f"top_pair_is_{r}{f}" for r in letters for f in letters])
    names.extend([
        "same_top1",
        "receiver_top1_score",
        "fusion_top1_score",
        "top1_score_diff_receiver_minus_fusion",
        "receiver_top1_log_prob",
        "fusion_top1_log_prob",
        "top1_log_prob_diff_receiver_minus_fusion",
        "receiver_top1_margin",
        "fusion_top1_margin",
        "top1_margin_diff_receiver_minus_fusion",
        "receiver_entropy",
        "fusion_entropy",
        "entropy_diff_fusion_minus_receiver",
        "receiver_prob_on_fusion_top1",
        "fusion_prob_on_receiver_top1",
        "receiver_top1_minus_receiver_prob_on_fusion_top1",
        "fusion_top1_minus_fusion_prob_on_receiver_top1",
        "kl_receiver_to_fusion",
        "kl_fusion_to_receiver",
        "js_receiver_fusion",
        "l1_receiver_fusion",
        "l2_receiver_fusion",
        "dot_receiver_fusion",
        "cosine_receiver_fusion",
        "receiver_sorted_top1_margin",
        "fusion_sorted_top1_margin",
    ])
    return names


def feature_names(feature_set: str = "basic38", num_options: int = 4) -> List[str]:
    if feature_set in {"basic", "basic38"}:
        return _basic_feature_names(num_options)
    if feature_set == "logpair_v1":
        return _logpair_feature_names(num_options)
    if feature_set == "compact_v1":
        return _compact_feature_names(num_options)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _parse_probs(row: Dict[str, str], key: str) -> np.ndarray:
    value = row.get(key)
    if not value:
        raise ValueError(f"Missing {key} in row {row.get('subject')}#{row.get('question_id')}")
    probs = np.asarray(json.loads(value), dtype=float)
    if probs.ndim != 1:
        raise ValueError(f"{key} must be 1D, got shape {probs.shape}")
    return probs


def _build_basic_features(receiver: np.ndarray, fusion: np.ndarray) -> np.ndarray:
    if receiver.shape != fusion.shape:
        raise ValueError(f"receiver/fusion prob shape mismatch: {receiver.shape} vs {fusion.shape}")

    num_options = receiver.shape[0]
    r_top = int(receiver.argmax())
    f_top = int(fusion.argmax())
    r_sorted = np.sort(receiver)
    f_sorted = np.sort(fusion)

    features: List[float] = []
    features.extend(receiver.tolist())
    features.extend(fusion.tolist())
    features.extend((receiver - fusion).tolist())
    features.extend((receiver / np.clip(fusion, 1e-12, None)).tolist())
    features.extend(_one_hot(r_top, num_options))
    features.extend(_one_hot(f_top, num_options))
    features.extend([
        float(receiver[r_top]),
        float(fusion[f_top]),
        float(receiver[r_top] - fusion[f_top]),
        _top1_margin(receiver),
        _top1_margin(fusion),
        _top1_margin(receiver) - _top1_margin(fusion),
        _entropy(receiver),
        _entropy(fusion),
        _entropy(fusion) - _entropy(receiver),
        float(r_top == f_top),
        float(receiver[f_top]),
        float(fusion[r_top]),
        float(receiver[r_top] - receiver[f_top]),
        float(fusion[f_top] - fusion[r_top]),
        float(r_sorted[-1] - r_sorted[-2]) if num_options >= 2 else 0.0,
        float(f_sorted[-1] - f_sorted[-2]) if num_options >= 2 else 0.0,
    ])
    return np.asarray(features, dtype=np.float32)


def _build_logpair_features(receiver: np.ndarray, fusion: np.ndarray) -> np.ndarray:
    if receiver.shape != fusion.shape:
        raise ValueError(f"receiver/fusion prob shape mismatch: {receiver.shape} vs {fusion.shape}")

    num_options = receiver.shape[0]
    r_top = int(receiver.argmax())
    f_top = int(fusion.argmax())
    r_sorted = np.sort(receiver)
    f_sorted = np.sort(fusion)
    log_r = _safe_log(receiver)
    log_f = _safe_log(fusion)
    top_pair = [0.0] * (num_options * num_options)
    top_pair[r_top * num_options + f_top] = 1.0
    dot = float(np.dot(receiver, fusion))
    norm = float(np.linalg.norm(receiver) * np.linalg.norm(fusion))

    features: List[float] = []
    features.extend(receiver.tolist())
    features.extend(fusion.tolist())
    features.extend((receiver - fusion).tolist())
    features.extend(np.abs(receiver - fusion).tolist())
    features.extend(log_r.tolist())
    features.extend(log_f.tolist())
    features.extend((log_r - log_f).tolist())
    features.extend(np.abs(log_r - log_f).tolist())
    features.extend(np.outer(receiver, fusion).reshape(-1).tolist())
    features.extend(_one_hot(r_top, num_options))
    features.extend(_one_hot(f_top, num_options))
    features.extend(top_pair)
    features.extend([
        float(receiver[r_top]),
        float(fusion[f_top]),
        float(receiver[r_top] - fusion[f_top]),
        _top1_margin(receiver),
        _top1_margin(fusion),
        _top1_margin(receiver) - _top1_margin(fusion),
        _entropy(receiver),
        _entropy(fusion),
        _entropy(fusion) - _entropy(receiver),
        float(r_top == f_top),
        float(receiver[f_top]),
        float(fusion[r_top]),
        float(receiver[r_top] - receiver[f_top]),
        float(fusion[f_top] - fusion[r_top]),
        _kl_divergence(receiver, fusion),
        _kl_divergence(fusion, receiver),
        _js_divergence(receiver, fusion),
        float(np.abs(receiver - fusion).sum()),
        float(np.linalg.norm(receiver - fusion)),
        dot,
        dot / norm if norm > 0.0 else 0.0,
        float(r_sorted[-1] - r_sorted[-2]) if num_options >= 2 else 0.0,
        float(f_sorted[-1] - f_sorted[-2]) if num_options >= 2 else 0.0,
    ])
    return np.asarray(features, dtype=np.float32)


def _build_compact_features(receiver: np.ndarray, fusion: np.ndarray) -> np.ndarray:
    if receiver.shape != fusion.shape:
        raise ValueError(f"receiver/fusion prob shape mismatch: {receiver.shape} vs {fusion.shape}")

    num_options = receiver.shape[0]
    r_top = int(receiver.argmax())
    f_top = int(fusion.argmax())
    r_sorted = np.sort(receiver)
    f_sorted = np.sort(fusion)
    log_r = _safe_log(receiver)
    log_f = _safe_log(fusion)
    top_pair = [0.0] * (num_options * num_options)
    top_pair[r_top * num_options + f_top] = 1.0
    dot = float(np.dot(receiver, fusion))
    norm = float(np.linalg.norm(receiver) * np.linalg.norm(fusion))

    features: List[float] = []
    features.extend(top_pair)
    features.extend([
        float(r_top == f_top),
        float(receiver[r_top]),
        float(fusion[f_top]),
        float(receiver[r_top] - fusion[f_top]),
        float(log_r[r_top]),
        float(log_f[f_top]),
        float(log_r[r_top] - log_f[f_top]),
        _top1_margin(receiver),
        _top1_margin(fusion),
        _top1_margin(receiver) - _top1_margin(fusion),
        _entropy(receiver),
        _entropy(fusion),
        _entropy(fusion) - _entropy(receiver),
        float(receiver[f_top]),
        float(fusion[r_top]),
        float(receiver[r_top] - receiver[f_top]),
        float(fusion[f_top] - fusion[r_top]),
        _kl_divergence(receiver, fusion),
        _kl_divergence(fusion, receiver),
        _js_divergence(receiver, fusion),
        float(np.abs(receiver - fusion).sum()),
        float(np.linalg.norm(receiver - fusion)),
        dot,
        dot / norm if norm > 0.0 else 0.0,
        float(r_sorted[-1] - r_sorted[-2]) if num_options >= 2 else 0.0,
        float(f_sorted[-1] - f_sorted[-2]) if num_options >= 2 else 0.0,
    ])
    return np.asarray(features, dtype=np.float32)


def build_features(row: Dict[str, str], feature_set: str = "basic38") -> np.ndarray:
    receiver = _parse_probs(row, "dual_receiver_probs")
    fusion = _parse_probs(row, "dual_fusion_probs")
    if feature_set in {"basic", "basic38"}:
        return _build_basic_features(receiver, fusion)
    if feature_set == "logpair_v1":
        return _build_logpair_features(receiver, fusion)
    if feature_set == "compact_v1":
        return _build_compact_features(receiver, fusion)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def load_rows(path: Path, feature_set: str) -> Tuple[List[Dict[str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: List[Dict[str, str]] = []
    features = []
    utility = []
    groups = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            features.append(build_features(row, feature_set=feature_set))
            receiver_ok = _is_correct(row, "dual_receiver_pred")
            fusion_ok = _is_correct(row, "dual_fusion_pred")
            utility.append(int(receiver_ok) - int(fusion_ok))
            groups.append(str(row.get("subject", "")))

    X = np.stack(features, axis=0)
    utility_arr = np.asarray(utility, dtype=np.int64)
    train_mask = utility_arr != 0
    y = (utility_arr[train_mask] > 0).astype(np.int64)
    return rows, X, utility_arr, train_mask, y, np.asarray(groups)


def make_model(model_name: str, class_weight):
    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                C=0.25,
                class_weight=class_weight,
                solver="lbfgs",
            ),
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=8,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=300,
            l2_regularization=1.0,
            random_state=42,
        )
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_threshold(
    rows: List[Dict[str, str]],
    utility: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    correct = 0
    help_count = 0
    harm_count = 0
    receiver_count = 0
    source_counts = Counter()
    subject_stats = defaultdict(lambda: [0, 0])

    for row, util, prob in zip(rows, utility, probabilities):
        choose_receiver = prob >= threshold
        source = "receiver" if choose_receiver else "fusion"
        source_counts[source] += 1
        receiver_count += int(choose_receiver)

        pred_key = "dual_receiver_pred" if choose_receiver else "dual_fusion_pred"
        ok = _is_correct(row, pred_key)
        correct += int(ok)
        if choose_receiver and util > 0:
            help_count += 1
        elif choose_receiver and util < 0:
            harm_count += 1

        subject = str(row.get("subject", ""))
        subject_stats[subject][0] += int(ok)
        subject_stats[subject][1] += 1

    total = len(rows)
    return {
        "threshold": float(threshold),
        "accuracy": correct / total if total else 0.0,
        "correct": int(correct),
        "total": int(total),
        "help": int(help_count),
        "harm": int(harm_count),
        "net": int(help_count - harm_count),
        "receiver_count": int(receiver_count),
        "selected_source_counts": dict(source_counts),
        "subjects": {
            subject: corr / cnt if cnt else 0.0
            for subject, (corr, cnt) in sorted(subject_stats.items())
        },
    }


def tune_threshold(rows, utility, probabilities, min_threshold: float = 0.0) -> Dict[str, object]:
    candidates = np.linspace(min_threshold, 1.0, 1001)
    best = None
    for threshold in candidates:
        metrics = evaluate_threshold(rows, utility, probabilities, float(threshold))
        key = (
            metrics["correct"],
            metrics["net"],
            -metrics["harm"],
            -metrics["receiver_count"],
        )
        if best is None or key > best[0]:
            best = (key, metrics)
    return best[1]


def _wandb_log_threshold_curve(
    wandb_run,
    rows,
    utility,
    probabilities,
    min_threshold: float,
    prefix: str,
    step_offset: int,
) -> None:
    if wandb_run is None:
        return

    for step, threshold in enumerate(np.linspace(min_threshold, 1.0, 101)):
        metrics = evaluate_threshold(rows, utility, probabilities, float(threshold))
        wandb_run.log(
            {
                f"{prefix}/threshold_step": step,
                f"{prefix}/threshold": float(threshold),
                f"{prefix}/accuracy": metrics["accuracy"],
                f"{prefix}/correct": metrics["correct"],
                f"{prefix}/help": metrics["help"],
                f"{prefix}/harm": metrics["harm"],
                f"{prefix}/net": metrics["net"],
                f"{prefix}/receiver_count": metrics["receiver_count"],
                f"{prefix}/receiver_rate": metrics["receiver_count"] / max(1, metrics["total"]),
            },
            step=step_offset + step,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", choices=["logreg", "rf", "hgb"], default="logreg")
    parser.add_argument("--cv", choices=["group", "stratified"], default="group")
    parser.add_argument(
        "--oof-scope",
        choices=["differential", "all"],
        default="differential",
        help=(
            "Rows to score out-of-fold for threshold tuning. 'differential' "
            "matches the original behavior and sets tie rows to zero. 'all' "
            "predicts every held-out row, which better matches inference."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--positive-class-weight", type=float, default=4.0)
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument(
        "--feature-set",
        choices=["basic38", "logpair_v1", "compact_v1"],
        default="basic38",
        help="Feature representation built from receiver/fusion option probabilities.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    rows, X, utility, train_mask, y, groups = load_rows(args.input_csv, args.feature_set)
    X_train = X[train_mask]
    groups_train = groups[train_mask]

    class_weight = {0: 1.0, 1: float(args.positive_class_weight)}
    model = make_model(args.model, class_weight)
    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                name=args.wandb_run_name,
                config={
                    "input_csv": str(args.input_csv),
                    "output_dir": str(args.output_dir),
                    "model": args.model,
                    "cv": args.cv,
                    "oof_scope": args.oof_scope,
                    "folds": args.folds,
                    "positive_class_weight": args.positive_class_weight,
                    "min_threshold": args.min_threshold,
                    "feature_set": args.feature_set,
                    "num_examples": len(rows),
                    "num_differential_train_examples": int(train_mask.sum()),
                    "feature_dim": int(X.shape[1]),
                    "class_weight": class_weight,
                },
            )
        except ImportError as exc:
            raise RuntimeError("wandb is not installed; install wandb or use --wandb-mode disabled") from exc

    oof_train_prob = np.zeros(X_train.shape[0], dtype=np.float64)
    oof_all_prob = np.zeros(X.shape[0], dtype=np.float64)
    fold_summaries = []
    if args.oof_scope == "differential":
        if args.cv == "group":
            splitter = GroupKFold(n_splits=args.folds)
            splits = splitter.split(X_train, y, groups_train)
        else:
            splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
            splits = splitter.split(X_train, y)

        for fold, (tr_idx, val_idx) in enumerate(splits):
            fold_model = make_model(args.model, class_weight)
            fold_model.fit(X_train[tr_idx], y[tr_idx])
            oof_train_prob[val_idx] = fold_model.predict_proba(X_train[val_idx])[:, 1]
            oof_all_prob[np.where(train_mask)[0][val_idx]] = oof_train_prob[val_idx]
            fold_auc = roc_auc_score(y[val_idx], oof_train_prob[val_idx])
            fold_summaries.append({
                "fold": fold,
                "num_train": int(len(tr_idx)),
                "num_val": int(len(val_idx)),
                "auc": float(fold_auc),
                "positive_rate": float(y[val_idx].mean()),
                "oof_scope": args.oof_scope,
            })
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "cv/fold": fold,
                        "cv/fold_auc": float(fold_auc),
                        "cv/fold_positive_rate": float(y[val_idx].mean()),
                        "cv/fold_num_train": int(len(tr_idx)),
                        "cv/fold_num_val": int(len(val_idx)),
                    },
                    step=fold,
                )
    else:
        if args.cv == "group":
            splitter = GroupKFold(n_splits=args.folds)
            splits = splitter.split(X, utility, groups)
        else:
            splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
            splits = splitter.split(X, utility)

        train_mask_indices = np.where(train_mask)[0]
        for fold, (tr_idx, val_idx) in enumerate(splits):
            tr_diff_idx = tr_idx[train_mask[tr_idx]]
            val_diff_idx = val_idx[train_mask[val_idx]]
            fold_y_train = (utility[tr_diff_idx] > 0).astype(np.int64)
            fold_model = make_model(args.model, class_weight)
            fold_model.fit(X[tr_diff_idx], fold_y_train)
            oof_all_prob[val_idx] = fold_model.predict_proba(X[val_idx])[:, 1]
            oof_train_prob = oof_all_prob[train_mask_indices]

            val_diff_prob = oof_all_prob[val_diff_idx]
            val_diff_y = (utility[val_diff_idx] > 0).astype(np.int64)
            fold_auc = (
                roc_auc_score(val_diff_y, val_diff_prob)
                if len(np.unique(val_diff_y)) > 1
                else float("nan")
            )
            fold_summaries.append({
                "fold": fold,
                "num_train": int(len(tr_diff_idx)),
                "num_val": int(len(val_idx)),
                "num_val_differential": int(len(val_diff_idx)),
                "auc": float(fold_auc),
                "positive_rate": float(val_diff_y.mean()) if len(val_diff_y) else 0.0,
                "oof_scope": args.oof_scope,
            })
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "cv/fold": fold,
                        "cv/fold_auc": float(fold_auc),
                        "cv/fold_positive_rate": float(val_diff_y.mean()) if len(val_diff_y) else 0.0,
                        "cv/fold_num_train": int(len(tr_diff_idx)),
                        "cv/fold_num_val": int(len(val_idx)),
                    },
                    step=fold,
                )

    cv_best = tune_threshold(rows, utility, oof_all_prob, min_threshold=args.min_threshold)
    for threshold in [0.5, 0.7, 0.8, 0.9, cv_best["threshold"]]:
        cv_best[f"metrics_at_{threshold:.3f}"] = evaluate_threshold(
            rows,
            utility,
            oof_all_prob,
            float(threshold),
        )

    model.fit(X_train, y)
    train_prob_all = model.predict_proba(X)[:, 1]
    train_best = tune_threshold(rows, utility, train_prob_all, min_threshold=args.min_threshold)

    fusion_correct = sum(_is_correct(row, "dual_fusion_pred") for row in rows)
    receiver_correct = sum(_is_correct(row, "dual_receiver_pred") for row in rows)
    utility_counts = Counter(utility.tolist())

    summary = {
        "input_csv": str(args.input_csv),
        "model": args.model,
        "cv": args.cv,
        "oof_scope": args.oof_scope,
        "folds": args.folds,
        "positive_class_weight": args.positive_class_weight,
        "feature_set": args.feature_set,
        "num_examples": len(rows),
        "num_differential_train_examples": int(train_mask.sum()),
        "utility_counts": {
            "receiver_better": int(utility_counts.get(1, 0)),
            "tie": int(utility_counts.get(0, 0)),
            "fusion_better": int(utility_counts.get(-1, 0)),
        },
        "fusion_accuracy": fusion_correct / len(rows),
        "fusion_correct": int(fusion_correct),
        "receiver_accuracy": receiver_correct / len(rows),
        "receiver_correct": int(receiver_correct),
        "feature_names": feature_names(args.feature_set, num_options=4),
        "folds": fold_summaries,
        "cv_auc": float(roc_auc_score(y, oof_train_prob)),
        "cv_best": cv_best,
        "train_best": train_best,
    }

    if wandb_run is not None:
        _wandb_log_threshold_curve(wandb_run, rows, utility, oof_all_prob, args.min_threshold, "cv_threshold", 1000)
        _wandb_log_threshold_curve(wandb_run, rows, utility, train_prob_all, args.min_threshold, "train_threshold", 2000)
        wandb_run.log(
            {
                "summary/fusion_accuracy": summary["fusion_accuracy"],
                "summary/receiver_accuracy": summary["receiver_accuracy"],
                "summary/cv_auc": summary["cv_auc"],
                "summary/cv_best_threshold": cv_best["threshold"],
                "summary/cv_best_accuracy": cv_best["accuracy"],
                "summary/cv_best_help": cv_best["help"],
                "summary/cv_best_harm": cv_best["harm"],
                "summary/cv_best_net": cv_best["net"],
                "summary/cv_best_receiver_count": cv_best["receiver_count"],
                "summary/train_best_threshold": train_best["threshold"],
                "summary/train_best_accuracy": train_best["accuracy"],
                "summary/train_best_help": train_best["help"],
                "summary/train_best_harm": train_best["harm"],
                "summary/train_best_net": train_best["net"],
                "summary/train_best_receiver_count": train_best["receiver_count"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.output_dir / f"fusion_receiver_correctness_detector_{args.model}_{timestamp}.joblib"
    summary_path = args.output_dir / f"fusion_receiver_correctness_detector_{args.model}_{timestamp}_summary.json"
    config_path = args.output_dir / f"fusion_receiver_correctness_detector_{args.model}_{timestamp}_config.json"

    joblib.dump(model, model_path)
    summary["model_path"] = str(model_path)
    summary["recommended_threshold"] = cv_best["threshold"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "threshold": cv_best["threshold"],
                "feature_source": "dual_receiver_probs + dual_fusion_probs",
                "feature_set": args.feature_set,
                "default_source": "fusion",
                "positive_action": "receiver",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model to {model_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved config to {config_path}")
    print(
        "fusion="
        f"{summary['fusion_correct']}/{summary['num_examples']} "
        f"({summary['fusion_accuracy'] * 100:.2f}%)"
    )
    print(
        "receiver="
        f"{summary['receiver_correct']}/{summary['num_examples']} "
        f"({summary['receiver_accuracy'] * 100:.2f}%)"
    )
    print(
        "differential labels: "
        f"receiver_better={summary['utility_counts']['receiver_better']} "
        f"fusion_better={summary['utility_counts']['fusion_better']} "
        f"tie={summary['utility_counts']['tie']}"
    )
    print(f"cv_auc={summary['cv_auc']:.4f}")
    print(
        "cv_best "
        f"threshold={cv_best['threshold']:.3f} "
        f"correct={cv_best['correct']}/{cv_best['total']} "
        f"acc={cv_best['accuracy'] * 100:.2f}% "
        f"help={cv_best['help']} harm={cv_best['harm']} "
        f"net={cv_best['net']:+d} receiver={cv_best['receiver_count']}"
    )
    print(
        "train_best "
        f"threshold={train_best['threshold']:.3f} "
        f"correct={train_best['correct']}/{train_best['total']} "
        f"acc={train_best['accuracy'] * 100:.2f}% "
        f"help={train_best['help']} harm={train_best['harm']} "
        f"net={train_best['net']:+d} receiver={train_best['receiver_count']}"
    )
    if wandb_run is not None:
        wandb_run.summary["model_path"] = str(model_path)
        wandb_run.summary["summary_path"] = str(summary_path)
        wandb_run.summary["config_path"] = str(config_path)
        wandb_run.finish()


if __name__ == "__main__":
    main()
