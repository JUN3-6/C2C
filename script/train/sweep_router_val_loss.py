#!/usr/bin/env python
"""
Run multiple router-training experiments and stop early when min val/loss
reaches a target threshold.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


VAL_LINE_RE = re.compile(
    r"epoch=(\d+).*?val_loss=([0-9.\-]+).*?val_mean_routed_gain=([0-9.\-]+)"
)


def _default_experiments():
    return [
        {"tag": "base_lr1e3", "lr": "1e-3"},
        {"tag": "lr8e4", "lr": "8e-4"},
        {"tag": "lr7e4", "lr": "7e-4"},
        {"tag": "lr5e4", "lr": "5e-4"},
        {"tag": "lr3e4", "lr": "3e-4"},
        {"tag": "lr8e4_warm003", "lr": "8e-4", "lr_warmup_ratio": "0.03"},
        {"tag": "lr8e4_warm01", "lr": "8e-4", "lr_warmup_ratio": "0.1"},
        {"tag": "lr8e4_min005", "lr": "8e-4", "lr_min_ratio": "0.05"},
        {"tag": "lr8e4_min02", "lr": "8e-4", "lr_min_ratio": "0.2"},
        {"tag": "lr8e4_linear", "lr": "8e-4", "lr_scheduler_type": "linear"},
        {"tag": "lr5e4_linear", "lr": "5e-4", "lr_scheduler_type": "linear"},
        {
            "tag": "lr5e4_ce8_r12",
            "lr": "5e-4",
            "gain_ce_only_epochs": "8",
            "gain_ramp_epochs": "12",
        },
        {
            "tag": "lr5e4_ce12_r12",
            "lr": "5e-4",
            "gain_ce_only_epochs": "12",
            "gain_ramp_epochs": "12",
        },
        {
            "tag": "lr5e4_ce10_r20",
            "lr": "5e-4",
            "gain_ce_only_epochs": "10",
            "gain_ramp_epochs": "20",
        },
        {
            "tag": "lr5e4_ce8_r20",
            "lr": "5e-4",
            "gain_ce_only_epochs": "8",
            "gain_ramp_epochs": "20",
        },
        {
            "tag": "lr5e4_ce12_r20",
            "lr": "5e-4",
            "gain_ce_only_epochs": "12",
            "gain_ramp_epochs": "20",
        },
        {
            "tag": "lr3e4_ce10_r20",
            "lr": "3e-4",
            "gain_ce_only_epochs": "10",
            "gain_ramp_epochs": "20",
        },
        {
            "tag": "lr3e4_warm01_r20",
            "lr": "3e-4",
            "lr_warmup_ratio": "0.1",
            "gain_ce_only_epochs": "10",
            "gain_ramp_epochs": "20",
        },
        {"tag": "lr5e4_wd1e4", "lr": "5e-4", "weight_decay": "1e-4"},
        {"tag": "lr5e4_wd5e4", "lr": "5e-4", "weight_decay": "5e-4"},
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--target-min-val-loss", type=float, default=0.6790)
    parser.add_argument("--wandb-project", default="C2C")
    parser.add_argument("--wandb-entity", default="june6-hanyang-university")
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[2]
    train_data = Path(args.train_data)
    val_data = Path(args.val_data)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    common = {
        "hidden_dim": "512",
        "token_mlp_layers": "2",
        "dropout": "0.0",
        "batch_size": "256",
        "epochs": "40",
        "weight_decay": "0.0",
        "action_loss_weight": "1.0",
        "gain_loss_weight": "0.02",
        "action_class_weight_mode": "inverse",
        "fuse_threshold": "0.5",
        "selection_temperature": "1.0",
        "best_metric": "loss",
        "lr_scheduler_type": "cosine",
        "lr_warmup_ratio": "0.05",
        "lr_min_ratio": "0.1",
        "gain_ce_only_epochs": "10",
        "gain_ramp_epochs": "10",
        "device": "cuda",
    }

    experiments = _default_experiments()[: args.max_runs]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "target_min_val_loss": args.target_min_val_loss,
        "max_runs": args.max_runs,
        "runs": [],
        "stopped_early": False,
        "best": None,
    }

    best_loss = float("inf")
    best_run = None

    print(f"[SWEEP] output_root={output_root}")
    print(f"[SWEEP] target_min_val_loss={args.target_min_val_loss}")

    for i, exp in enumerate(experiments, start=1):
        cfg = dict(common)
        cfg.update(exp)
        tag = cfg.pop("tag")

        run_dir = output_root / f"run_{i:02d}_{tag}" / "router"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"val_loss_sweep_gain002_{timestamp}_{i:02d}_{tag}"
        log_path = run_dir / "sweep_train.log"

        cmd = [
            sys.executable,
            "script/train/train_router.py",
            "--train-data",
            str(train_data),
            "--val-data",
            str(val_data),
            "--output-dir",
            str(run_dir),
            "--hidden-dim",
            cfg["hidden_dim"],
            "--token-mlp-layers",
            cfg["token_mlp_layers"],
            "--dropout",
            cfg["dropout"],
            "--batch-size",
            cfg["batch_size"],
            "--epochs",
            cfg["epochs"],
            "--lr",
            cfg["lr"],
            "--weight-decay",
            cfg["weight_decay"],
            "--lr-scheduler-type",
            cfg["lr_scheduler_type"],
            "--lr-warmup-ratio",
            cfg["lr_warmup_ratio"],
            "--lr-min-ratio",
            cfg["lr_min_ratio"],
            "--action-loss-weight",
            cfg["action_loss_weight"],
            "--gain-loss-weight",
            cfg["gain_loss_weight"],
            "--gain-ce-only-epochs",
            cfg["gain_ce_only_epochs"],
            "--gain-ramp-epochs",
            cfg["gain_ramp_epochs"],
            "--action-class-weight-mode",
            cfg["action_class_weight_mode"],
            "--fuse-threshold",
            cfg["fuse_threshold"],
            "--selection-temperature",
            cfg["selection_temperature"],
            "--best-metric",
            cfg["best_metric"],
            "--device",
            cfg["device"],
            "--wandb",
            "--wandb-project",
            args.wandb_project,
            "--wandb-entity",
            args.wandb_entity,
            "--wandb-mode",
            args.wandb_mode,
            "--wandb-run-name",
            run_name,
            "--wandb-tag",
            "sweep-val-loss",
            "--wandb-tag",
            "gain002",
        ]

        print(f"\n[RUN {i:02d}/{args.max_runs}] {tag}")
        min_val_loss = float("inf")
        best_gain = float("-inf")
        best_epoch = None
        status = "ok"

        with log_path.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                m = VAL_LINE_RE.search(line)
                if m:
                    epoch = int(m.group(1))
                    val_loss = float(m.group(2))
                    val_gain = float(m.group(3))
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        best_epoch = epoch
                    if val_gain > best_gain:
                        best_gain = val_gain
                    if epoch % 5 == 0:
                        print(
                            f"  epoch={epoch:02d} "
                            f"min_val_loss_so_far={min_val_loss:.4f} "
                            f"max_gain_so_far={best_gain:.4f}"
                        )
            ret = proc.wait()

        if ret != 0:
            status = f"failed({ret})"

        if min_val_loss < best_loss:
            best_loss = min_val_loss
            best_run = {
                "index": i,
                "tag": tag,
                "min_val_loss": min_val_loss,
                "best_epoch_by_val_loss": best_epoch,
                "max_val_mean_routed_gain": best_gain,
                "output_dir": str(run_dir),
                "log_path": str(log_path),
            }

        run_result = {
            "index": i,
            "tag": tag,
            "status": status,
            "min_val_loss": None if min_val_loss == float("inf") else min_val_loss,
            "best_epoch_by_val_loss": best_epoch,
            "max_val_mean_routed_gain": None if best_gain == float("-inf") else best_gain,
            "output_dir": str(run_dir),
            "log_path": str(log_path),
        }
        summary["runs"].append(run_result)
        summary["best"] = best_run

        print(
            f"[RESULT {i:02d}] status={status} "
            f"min_val_loss={run_result['min_val_loss']} "
            f"best_epoch={best_epoch} "
            f"max_gain={run_result['max_val_mean_routed_gain']}"
        )

        if status == "ok" and min_val_loss <= args.target_min_val_loss:
            print(
                "[EARLY STOP] target reached: "
                f"min_val_loss={min_val_loss:.4f} <= {args.target_min_val_loss}"
            )
            summary["stopped_early"] = True
            break

    summary_path = output_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[SWEEP DONE]")
    print(f"summary_path={summary_path}")
    print(f"best={best_run}")


if __name__ == "__main__":
    main()
