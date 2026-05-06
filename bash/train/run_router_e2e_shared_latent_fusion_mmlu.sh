#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-/home/june/workspace/C2C_routing/local/checkpoints/0.6B+0.5B_MMLU_15k/config.json}
MIX_CONFIGS=${MIX_CONFIGS:-}
BANK_DIR=${BANK_DIR:-/home/june/workspace/C2C_routing/local/checkpoints/0.6B+0.5B_MMLU_15k/final}
BANK_DIRS=${BANK_DIRS:-}
OUTPUT_DIR=${OUTPUT_DIR:-/home/june/workspace/C2C_routing/local/router_runs/e2e_shared_latent_fusion_mmlu_option_ce_v1}

NUM_SAMPLES=${NUM_SAMPLES:-1024}
DATA_POOL_SAMPLES=${DATA_POOL_SAMPLES:-}
EVAL_SAMPLES=${EVAL_SAMPLES:-128}
MAX_LENGTH=${MAX_LENGTH:-}
EPOCHS=${EPOCHS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-0.00005}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
EVAL_EVERY=${EVAL_EVERY:-20}
LOG_EVERY=${LOG_EVERY:-5}
FUSION_COST_WEIGHT=${FUSION_COST_WEIGHT:-0.0}
TRAINING_OBJECTIVE=${TRAINING_OBJECTIVE:-action_ce_oracle}
LABEL_METRIC=${LABEL_METRIC:-option_token_ce}
NO_SKIP_ROUTING=${NO_SKIP_ROUTING:-0}
ACTION_CLASS_WEIGHT_MODE=${ACTION_CLASS_WEIGHT_MODE:-inverse}
ACTION_CLASS_WEIGHT_SCAN_SAMPLES=${ACTION_CLASS_WEIGHT_SCAN_SAMPLES:-0}
ACTION_CLASS_MAX_WEIGHT=${ACTION_CLASS_MAX_WEIGHT:-10.0}
ACTION_CLASS_SKIP_MULTIPLIER=${ACTION_CLASS_SKIP_MULTIPLIER:-1.0}
ORACLE_CE_MARGIN=${ORACLE_CE_MARGIN:-0.0}
ORACLE_CE_MARGIN_QUANTILE=${ORACLE_CE_MARGIN_QUANTILE:-}
DELTA_HIST_BINS=${DELTA_HIST_BINS:-21}

LATENT_ENCODER_DIM=${LATENT_ENCODER_DIM:-1024}
LATENT_SHARED_DIM=${LATENT_SHARED_DIM:-1024}
LATENT_NUM_HEADS=${LATENT_NUM_HEADS:-16}
LATENT_FFN_MULT=${LATENT_FFN_MULT:-4.0}
LATENT_DROPOUT=${LATENT_DROPOUT:-0.0}
LATENT_POOLING=${LATENT_POOLING:-last}
LATENT_MAX_ENCODER_TOKENS=${LATENT_MAX_ENCODER_TOKENS:-0}
ROUTER_FEATURE_SOURCE=${ROUTER_FEATURE_SOURCE:-shared_latent_fusion}

ROUTER_HIDDEN_DIM=${ROUTER_HIDDEN_DIM:-512}
ROUTER_LAYERS=${ROUTER_LAYERS:-2}
ROUTER_DROPOUT=${ROUTER_DROPOUT:-0.1}
ROUTER_INIT_SCALE=${ROUTER_INIT_SCALE:-1.0}
ROUTER_DTYPE=${ROUTER_DTYPE:-}
LOAD_BALANCE_LOSS_WEIGHT=${LOAD_BALANCE_LOSS_WEIGHT:-0.0}
LOAD_BALANCE_LOSS_TYPE=${LOAD_BALANCE_LOSS_TYPE:-mse}
PROJECTOR_TRAIN_MODE=${PROJECTOR_TRAIN_MODE:-frozen}

WANDB_MODE=${WANDB_MODE:-online}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-e2e_shared_latent_fusion_mmlu_option_ce_v1_${NUM_SAMPLES}_ep${EPOCHS}}

EXTRA_ARGS=()
if [[ -n "$ORACLE_CE_MARGIN_QUANTILE" ]]; then
  EXTRA_ARGS+=(--oracle-ce-margin-quantile "$ORACLE_CE_MARGIN_QUANTILE")
fi
if [[ -n "$ROUTER_DTYPE" ]]; then
  EXTRA_ARGS+=(--router-dtype "$ROUTER_DTYPE")
fi
if [[ -n "$DATA_POOL_SAMPLES" ]]; then
  EXTRA_ARGS+=(--data-pool-samples "$DATA_POOL_SAMPLES")
fi
if [[ -n "$MAX_LENGTH" ]]; then
  EXTRA_ARGS+=(--max-length "$MAX_LENGTH")
fi
if [[ -n "$MIX_CONFIGS" ]]; then
  IFS=':' read -r -a MIX_CONFIG_ARRAY <<< "$MIX_CONFIGS"
  for mix_config in "${MIX_CONFIG_ARRAY[@]}"; do
    EXTRA_ARGS+=(--mix-data-config "$mix_config")
  done
fi
if [[ "$NO_SKIP_ROUTING" == "1" || "$NO_SKIP_ROUTING" == "true" ]]; then
  EXTRA_ARGS+=(--no-skip-routing)
fi

PROJECTOR_BANK_ARGS=()
if [[ -n "$BANK_DIRS" ]]; then
  IFS=':' read -r -a BANK_DIR_ARRAY <<< "$BANK_DIRS"
  for bank_dir in "${BANK_DIR_ARRAY[@]}"; do
    PROJECTOR_BANK_ARGS+=(--projector-bank-dir "$bank_dir")
  done
else
  PROJECTOR_BANK_ARGS+=(--projector-bank-dir "$BANK_DIR")
fi

python -u script/train/train_router_e2e_softgate.py \
  --config "$CONFIG" \
  "${PROJECTOR_BANK_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --threshold-sweep-samples "$EVAL_SAMPLES" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --eval-every "$EVAL_EVERY" \
  --log-every "$LOG_EVERY" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --fusion-cost-weight "$FUSION_COST_WEIGHT" \
  --training-objective "$TRAINING_OBJECTIVE" \
  --action-class-weight-mode "$ACTION_CLASS_WEIGHT_MODE" \
  --action-class-weight-scan-samples "$ACTION_CLASS_WEIGHT_SCAN_SAMPLES" \
  --action-class-max-weight "$ACTION_CLASS_MAX_WEIGHT" \
  --action-class-skip-multiplier "$ACTION_CLASS_SKIP_MULTIPLIER" \
  --oracle-ce-margin "$ORACLE_CE_MARGIN" \
  --delta-hist-bins "$DELTA_HIST_BINS" \
  --label-metric "$LABEL_METRIC" \
  --router-feature-source "$ROUTER_FEATURE_SOURCE" \
  --router-hidden-dim "$ROUTER_HIDDEN_DIM" \
  --router-layers "$ROUTER_LAYERS" \
  --router-dropout "$ROUTER_DROPOUT" \
  --router-init-scale "$ROUTER_INIT_SCALE" \
  --load-balance-loss-weight "$LOAD_BALANCE_LOSS_WEIGHT" \
  --load-balance-loss-type "$LOAD_BALANCE_LOSS_TYPE" \
  --latent-encoder-dim "$LATENT_ENCODER_DIM" \
  --latent-shared-dim "$LATENT_SHARED_DIM" \
  --latent-num-heads "$LATENT_NUM_HEADS" \
  --latent-ffn-mult "$LATENT_FFN_MULT" \
  --latent-dropout "$LATENT_DROPOUT" \
  --latent-pooling "$LATENT_POOLING" \
  --latent-max-encoder-tokens "$LATENT_MAX_ENCODER_TOKENS" \
  --projector-train-mode "$PROJECTOR_TRAIN_MODE" \
  --device cuda:0 \
  --wandb \
  --wandb-project C2C \
  --wandb-entity june6-hanyang-university \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  "${EXTRA_ARGS[@]}"
