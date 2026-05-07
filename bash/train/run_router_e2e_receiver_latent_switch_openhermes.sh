#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-/home/june/workspace/C2C_routing/local/checkpoints/0.6+0.5B_C2C_general_again/config.json}
BANK_DIR=${BANK_DIR:-/home/june/workspace/C2C_routing/local/checkpoints/0.6+0.5B_C2C_general_again/final}
OUTPUT_DIR=${OUTPUT_DIR:-/home/june/workspace/C2C_routing/local/router_runs/e2e_receiver_latent_switch_openhermes_4expert_projector_full_v1}

NUM_SAMPLES=${NUM_SAMPLES:-2048}
DATA_POOL_SAMPLES=${DATA_POOL_SAMPLES:-20000}
EVAL_SAMPLES=${EVAL_SAMPLES:-128}
MAX_LENGTH=${MAX_LENGTH:-2048}
EPOCHS=${EPOCHS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-0.00002}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
EVAL_EVERY=${EVAL_EVERY:-50}
LOG_EVERY=${LOG_EVERY:-10}

# Switch-style routing: hard top-1 forward, soft router gradient, plus aux load balance.
TRAINING_OBJECTIVE=${TRAINING_OBJECTIVE:-task_ce_straight_through}
LABEL_METRIC=${LABEL_METRIC:-response_ce}
FUSION_COST_WEIGHT=${FUSION_COST_WEIGHT:-0.0}
LOAD_BALANCE_LOSS_WEIGHT=${LOAD_BALANCE_LOSS_WEIGHT:-0.01}
LOAD_BALANCE_LOSS_TYPE=${LOAD_BALANCE_LOSS_TYPE:-switch}
ROUTER_INIT_SCALE=${ROUTER_INIT_SCALE:-0.1}
NUM_EXPERTS=${NUM_EXPERTS:-4}
EXPERT_DROPOUT=${EXPERT_DROPOUT:-0.4}

ROUTER_FEATURE_SOURCE=${ROUTER_FEATURE_SOURCE:-shared_latent_receiver}
LATENT_ENCODER_DIM=${LATENT_ENCODER_DIM:-1024}
LATENT_SHARED_DIM=${LATENT_SHARED_DIM:-1024}
LATENT_NUM_HEADS=${LATENT_NUM_HEADS:-16}
LATENT_FFN_MULT=${LATENT_FFN_MULT:-4.0}
LATENT_DROPOUT=${LATENT_DROPOUT:-0.0}
LATENT_POOLING=${LATENT_POOLING:-last}
LATENT_MAX_ENCODER_TOKENS=${LATENT_MAX_ENCODER_TOKENS:-128}

ROUTER_HIDDEN_DIM=${ROUTER_HIDDEN_DIM:-512}
ROUTER_LAYERS=${ROUTER_LAYERS:-2}
ROUTER_DROPOUT=${ROUTER_DROPOUT:-0.1}
ROUTER_DTYPE=${ROUTER_DTYPE:-float32}

PROJECTOR_TRAIN_MODE=${PROJECTOR_TRAIN_MODE:-full}
DEVICE=${DEVICE:-cuda:0}
PROJECTOR_DEVICES=${PROJECTOR_DEVICES:-$DEVICE}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-e2e_receiver_latent_switch_openhermes_4expert_projector_full_${NUM_SAMPLES}_ep${EPOCHS}}

EXTRA_ARGS=()
if [[ -n "$DATA_POOL_SAMPLES" ]]; then
  EXTRA_ARGS+=(--data-pool-samples "$DATA_POOL_SAMPLES")
fi
if [[ -n "$MAX_LENGTH" ]]; then
  EXTRA_ARGS+=(--max-length "$MAX_LENGTH")
fi
if [[ -n "$ROUTER_DTYPE" ]]; then
  EXTRA_ARGS+=(--router-dtype "$ROUTER_DTYPE")
fi

python -u script/train/train_router_e2e_softgate.py \
  --config "$CONFIG" \
  --projector-bank-dir "$BANK_DIR" \
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
  --warmup-ratio "$WARMUP_RATIO" \
  --fusion-cost-weight "$FUSION_COST_WEIGHT" \
  --no-skip-routing \
  --num-experts "$NUM_EXPERTS" \
  --training-objective "$TRAINING_OBJECTIVE" \
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
  --expert-dropout "$EXPERT_DROPOUT" \
  --device "$DEVICE" \
  --projector-devices "$PROJECTOR_DEVICES" \
  --wandb \
  --wandb-project C2C \
  --wandb-entity june6-hanyang-university \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  "${EXTRA_ARGS[@]}"
