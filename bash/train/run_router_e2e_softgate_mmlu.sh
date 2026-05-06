#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-/home/june/workspace/C2C_routing/local/checkpoints/0.6B+0.5B_MMLU_15k/config.json}
BANK_DIR=${BANK_DIR:-/home/june/workspace/C2C_routing/local/checkpoints/0.6B+0.5B_MMLU_15k/final}
OUTPUT_DIR=${OUTPUT_DIR:-/home/june/workspace/C2C_routing/local/router_runs/e2e_softgate_mmlu_option_frozen_cost001}

NUM_SAMPLES=${NUM_SAMPLES:-512}
EVAL_SAMPLES=${EVAL_SAMPLES:-64}
EPOCHS=${EPOCHS:-1}
FUSION_COST_WEIGHT=${FUSION_COST_WEIGHT:-0.01}
PROJECTOR_TRAIN_MODE=${PROJECTOR_TRAIN_MODE:-frozen}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-e2e_softgate_mmlu_option_${PROJECTOR_TRAIN_MODE}_cost${FUSION_COST_WEIGHT}_${NUM_SAMPLES}}

python -u script/train/train_router_e2e_softgate.py \
  --config "$CONFIG" \
  --projector-bank-dir "$BANK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --threshold-sweep-samples "$EVAL_SAMPLES" \
  --epochs "$EPOCHS" \
  --gradient-accumulation-steps 8 \
  --eval-every 20 \
  --log-every 10 \
  --lr 0.0001 \
  --weight-decay 0.01 \
  --fusion-cost-weight "$FUSION_COST_WEIGHT" \
  --label-metric option_token_ce \
  --input-standardization \
  --feature-stat-samples 128 \
  --projector-train-mode "$PROJECTOR_TRAIN_MODE" \
  --device cuda:0 \
  --wandb \
  --wandb-project C2C \
  --wandb-entity june6-hanyang-university \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name "$WANDB_RUN_NAME"
