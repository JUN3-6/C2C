# C2C Router Project Overview

이 문서는 현재 코드 기준으로,  
`KV cache -> router feature -> action 결정(skip/bank)` 흐름을 정확히 설명합니다.

핵심은 다음 5가지입니다.

1. 현재 router 입력 feature는 **hidden feature(현재 실험 기준 1408차원)**입니다.
2. label은 `improvements = ce_receiver - ce_fusions` 기반으로 생성됩니다.
3. router는 이진+선택 2헤드가 아니라 **멀티클래스 action head**로 학습합니다.
4. action 클래스는 `0=skip`, `1..K=bank_(class-1)` 입니다.
5. action CE에는 **optional weighted CE(클래스 불균형 보정)**를 적용할 수 있습니다.

---

## 1. 현재 파이프라인 요약

메인 실행 스크립트:

- `/home/june/workspace/C2C_routing/bash/train/run_router_full_pipeline.sh`
- `/home/june/workspace/C2C_routing/bash/train/run_router_full_pipeline_mmlu_option.sh`

실행 순서:

1. shard별 라벨/feature 생성  
   `script/train/generate_router_labels.py`
2. shard 통합  
   `script/train/build_router_dataset.py`
3. train/val split
4. router 학습  
   `script/train/train_router.py`

참고:

- 두 파이프라인 스크립트 모두 weighted CE 옵션을 그대로 전달할 수 있습니다.

---

## 2. KV cache가 router 입력으로 들어가는 방식

### 2.1 KV cache 원형 텐서

router feature 추출 함수는 KV 텐서가 아래 shape라고 가정합니다.

- `(B, H, N, D)`  
  - `B`: batch
  - `H`: num heads (KV heads)
  - `N`: token 길이
  - `D`: head dim

관련 코드:

- `rosetta/model/router.py` 의 `_mean_token_hidden`

### 2.2 prefill 구간에서 cache 수집

라벨 생성 시에는 `kv_cache_index[:-1]`만 사용합니다.

- 즉, **final response section은 제외**하고 prefill section들만 누적 forward하여
- base/teacher dynamic cache를 만든 뒤 feature를 추출합니다.

관련 코드:

- `script/train/generate_router_labels.py` 의 `compute_prefill_caches`

### 2.3 feature 가공 수식

`extract_query_feature` 내부 흐름:

1. 각 layer에서 key/value를 token 축 평균  
   `key.mean(dim=2)`, `value.mean(dim=2)` -> `(B, H, D)`
2. key/value 평균을 다시 평균  
   `0.5 * (key_mean + value_mean)` -> `(B, H, D)`
3. layer 평균  
   여러 layer를 평균 -> `base_hidden`, `source_hidden` 각각 `(B, H, D)`
4. flatten  
   `base_flat`, `source_flat`
5. 차이 항 생성  
   `common_dim = min(len(base_flat), len(source_flat))`  
   `diff = source_common - base_common`
6. 최종 concat

```text
feature = [base_flat, source_flat, diff, abs(diff)]
```

---

## 3. 현재 feature 차원: 1408

현재 주력 설정(`Qwen3-0.6B` receiver, `Qwen2.5-0.5B-Instruct` sharer)에서:

- base hidden: `8 * 128 = 1024`
- source hidden: `2 * 64 = 128`
- diff: `128`
- abs(diff): `128`

그래서:

```text
input_dim = 1024 + 128 + 128 + 128 = 1408
```

중요:

- `train_router.py`는 dataset의 `pooled_feature.shape[-1]`을 읽어 `input_dim`을 맞춰 학습합니다.

---

## 4. 라벨 생성 로직

샘플별로 아래를 계산합니다.

1. `ce_receiver` (receiver-only CE)
2. `ce_fusions[b]` (bank별 fusion CE)
3. `improvements[b] = ce_receiver - ce_fusions[b]`

`skip_margin` 이하 improvement는 skip 후보로 처리합니다.

현재 저장되는 타깃:

- `binary_target`: `0=skip`, `1=fuse`
- `bank_target`: skip이면 `-1`, fuse면 bank index
- `action_target`: `0=skip`, fuse면 `bank_index+1`

관련 코드:

- `script/train/generate_router_labels.py` (`compute_skip_and_bank_labels`)
- `script/train/generate_router_labels_option_ce.py` (`compute_skip_and_bank_labels`)
- `script/train/build_router_dataset.py` (`_build_labels`)

---

## 5. Router 모델 구조 (멀티클래스)

현재 `SimpleKVRouter` 출력은 `action_logits` 하나입니다.

- 클래스 수: `num_actions = num_banks + 1`
- 클래스 의미:
  - `0`: skip
  - `1..K`: bank `0..K-1`

보조 파생값:

- `selection_logits = action_logits[:, 1:]`
- `fuse_probability = 1 - P(skip)`  
- `binary_logits`는 `fuse vs skip` log-odds로 파생 계산

관련 코드:

- `rosetta/model/router.py` (`RouterOutput`, `SimpleKVRouter`)

---

## 6. 학습 objective (현재)

`train_router.py` total loss:

```text
total_loss
  = action_loss_weight * action_ce
  + gain_loss_weight * gain_loss
```

- `action_ce = CE(action_logits, action_target, weight=action_class_weights?)`
- `gain_loss = -mean(expected_routed_gain)`

```text
expected_routed_gain
  = sum(softmax(action_logits / T) * action_gain)

action_gain = [0, improvements[0], improvements[1], ..., improvements[K-1]]
```

즉, 단순 action 분류 정확도뿐 아니라  
실제 CE gain 회수도 같이 최적화합니다.

### 6.1 Weighted CE (클래스 불균형 보정)

멀티클래스 action 분포가 치우칠 때(예: skip 희소),  
`action_class_weights`를 계산해 CE에 적용할 수 있습니다.

지원 모드:

- `none`: 가중치 미적용
- `inverse`: 클래스 빈도 역수 기반
- `effective_num`: class-balanced weighting (`beta` 사용)

추가 안정화:

- 가중치 max clip (`--action-class-max-weight`)
- skip 클래스 추가 스케일 (`--action-class-skip-multiplier`)
- 가중치 평균 재정규화(스케일 안정화)

관련 코드:

- `script/train/train_router.py`
  - `_compute_action_class_weights`
  - `_compute_losses` (`F.cross_entropy(..., weight=...)`)

참고:

- `--selection-loss-weight`는 하위 호환 alias로만 받고, 내부에서 `--action-loss-weight`로 매핑됩니다.
- weighted CE 옵션:
  - `--action-class-weight-mode {none,inverse,effective_num}`
  - `--action-class-effective-num-beta <float>`
  - `--action-class-max-weight <float>`
  - `--action-class-skip-multiplier <float>`

---

## 7. Validation 지표

`train_router.py`에서 아래를 계산/로깅합니다.

- `val/loss`
- `val/action_loss`
- `val/gain_loss`
- `val/action_acc`
- `val/binary_acc`
- `val/bank_acc`
- `val/mean_routed_gain`
- `val/mean_oracle_gain`
- `val/gain_capture`
- `val/harm_rate`
- `val/fuse_rate`, `val/skip_rate`
- `val/wrong_bank_gain_delta`

best checkpoint metric은 `--best-metric`으로 선택합니다.  
`auto`일 때는 improvements가 있으면 `mean_routed_gain` 우선입니다.

---

## 8. Inference에서 router가 언제 호출되는가

관련 핵심 코드:

- `rosetta/model/wrapper.py`

현재 routing 경로:

- prefill section 루프에서 router를 호출하고 bank를 적용합니다.
- routing 모드에서 `include_response=false`면 decode-time 지속 fusion은 하지 않습니다.
- `include_response=true`일 때만 last-section hook 기반 경로가 추가됩니다.

현재 `predict()`의 최종 결정은 one-shot action 정책입니다.

- `selected_action = argmax(action_logits)`
- `selected_action == 0` 이면 skip
- `selected_action > 0` 이면 fuse, 선택 bank는 `selected_action - 1`

즉, **inference-time action 결정에 fuse threshold는 사용하지 않습니다**.  
(`fuse_threshold` 인자는 호출 시그니처/과거 호환을 위해 남아 있으며, 학습 validation metric 계산에만 의미가 있습니다.)

---

## 9. 저장 산출물

### shard 출력 (`generate_router_labels*.py`)

각 예시에 다음이 저장됩니다.

- `pooled_feature`
- `ce_receiver`
- `ce_fusions`
- `improvements`
- `binary_target`
- `bank_target`
- `action_target`
- `best_action`
- 기타 metadata (`sample_idx`, `sample_id`, `messages`, `num_response_tokens`)

### 통합 데이터셋 (`build_router_dataset.py`)

- `router_dataset_full.pt`
- `router_dataset_train.pt`
- `router_dataset_val.pt`
- `router_dataset_split_meta.pt`

### 학습 결과 (`train_router.py`)

- `router/router.json`
- `router/router.pt`
- logs에 train action count / class weights 출력

---

## 10. Legacy 체크포인트 호환

구형 router(`binary_head + selection_head`) 체크포인트를 로드하면,  
현재 `action_head` 형식으로 자동 마이그레이션합니다.

관련 코드:

- `rosetta/model/router.py` (`_migrate_legacy_state_dict`, `load_state_dict`)

주의:

- 자동 마이그레이션은 호환용입니다.  
  최종 성능 비교/배포는 멀티클래스 구조로 재학습한 체크포인트를 권장합니다.

---

## 11. 현재 주의사항

1. **입력 차원 불일치**
   - router checkpoint의 입력 차원과 runtime feature 차원이 다르면 즉시 shape mismatch가 납니다.
2. **bank 순서 불변성**
   - 학습 시 bank index 순서와 inference 시 `projector_bank_dirs` 순서는 반드시 동일해야 합니다.
3. **W&B**
   - 기본 pipeline에서는 preprocessing 단계가 아니라 training 단계에서 W&B run이 뜹니다.
4. **클래스 가중치 과도 설정**
   - 희소 클래스 가중치를 너무 크게 두면 학습이 불안정해질 수 있어 `max_weight`/`skip_multiplier` 튜닝이 필요합니다.

---

## 12. 빠른 체크리스트

- 내가 쓰는 router 입력 차원이 현재 feature와 맞는가?  
  - `router.pt`의 `feature_encoder.0.weight.shape[1]` 확인 (예: 1408)
- eval yaml의 `router_dir`가 의도한 checkpoint를 가리키는가?
- eval yaml의 `projector_bank_dirs` 순서가 학습 때와 동일한가?
- `include_response` 설정이 의도( decode-time fusion 여부 )와 맞는가?
- one-shot action 정책인지 확인했는가? (`selected_action = argmax(action_logits)`)
- weighted CE를 켰다면 action count/weights 출력이 의도와 맞는가?
- 파이프라인 실행 시 weighted CE 옵션을 함께 넘겼는가?
