# C2C Routing 구현 명세

## 1. 목표

`wrapper.py`를 확장해서, prefill 이후 fusion 직전에 routing을 수행한다.

routing의 목적은 두 가지다.

1. 이번 query에서 fusion을 할지 말지 결정한다.
2. fusion을 한다면 여러 projector candidate bank 중 어떤 bank를 사용할지 결정한다.

최종적으로는 receiver와 sharer의 prefill KV cache feature를 보고,
`skip` 또는 `K개 projector bank 중 하나`를 hard top-1으로 선택하는 구조를 만든다.


## 2. 이번 버전에서 확정된 범위

### 포함

- `single sharer only`
- `query당 1회 routing`
- routing decision은 `모든 target layer에 공통`
- `binary head + selection head` 분리
- offline label 생성 파이프라인
- router 전용 학습 파이프라인
- projector bank를
  - 이미 학습된 checkpoint 묶음으로 로드하는 방식
  - bank별 독립 학습으로 만드는 방식
  둘 다 지원

### 제외

- multi-sharer routing
- projector bank와 router의 joint co-training
- 기존 entropy gate 유지

기존 entropy gate는 새 routing probe로 완전히 대체한다.


## 3. Projector Bank 정의

`bank`는 모든 target layer에 대한 projector를 묶은 하나의 `layer-wise projector set`이다.

즉 target layer가 `L`개라면, bank 하나는 `L`개의 projector로 이루어진다.

- bank 0 = `[P0^0, P0^1, ..., P0^{L-1}]`
- bank 1 = `[P1^0, P1^1, ..., P1^{L-1}]`
- ...
- bank K-1 = `[P{K-1}^0, P{K-1}^1, ..., P{K-1}^{L-1}]`

여기서 `Pb^t`는 `bank b`에 속한 `target layer t용 projector`를 뜻한다.

router는 query마다 projector를 layer별로 따로 고르지 않는다.
대신 `bank 하나`를 통째로 고르고, 모든 target layer는 그 bank에 속한 자기 layer projector를 사용한다.

즉 이번 버전의 선택 단위는 `개별 layer projector`가 아니라 `projector set(bank)`다.

### 기존 C2C와의 비교

#### K = 1

- projector set이 1개뿐이다
- action은 사실상 `{skip, bank_0}` 이다
- `bank_0`를 고르면 모든 layer가 기존 C2C처럼 자기 layer용 projector를 사용한다

즉 `K=1`은 기존 C2C의 projector 구조를 유지하면서,
query마다 `fuse할지(skip/fuse)`만 추가로 결정하는 형태다.

#### K = 2

- projector set이 2개 있다
- action은 `{skip, bank_0, bank_1}` 이다
- `bank_0`를 고르면 모든 layer가 set 0을 사용한다
- `bank_1`를 고르면 모든 layer가 set 1을 사용한다

즉 `K=2`는 query마다

- fusion을 아예 하지 않을지
- projector set A를 쓸지
- projector set B를 쓸지

를 고르는 구조다.

정리하면,

- 기존 C2C: projector set이 1개이고 항상 그 set을 사용
- `K=1`: projector set이 1개이고 query마다 `skip` 또는 `그 set 사용`을 결정
- `K=2`: projector set이 2개이고 query마다 `skip / set 0 / set 1` 중 하나를 결정


## 4. Routing 입력과 모델 방향

routing 입력은 receiver와 sharer의 prefill KV cache feature다.

요구사항은 다음과 같다.

- 여러 layer를 모두 본다
- sequence length를 고정 차원으로 두지 않는다
- 각 token 위치에 같은 linear/MLP를 공유 적용한다
- token pooling과 layer aggregation을 거쳐 query-level representation을 만든다

첫 버전의 probe/router는 간단하게 시작한다.

- token-wise shared linear 기반
- multi-layer pooling 포함
- query-level decision 출력

router 모듈은 projector와 분리된 별도 파일로 만든다.
`wrapper`에는 routing 추론 로직만 들어가고, router 학습 코드는 별도 script로 둔다.


## 5. Label 정의

핵심 지표는 `response CE improvement`다.

- `CE_receiver`: receiver 단독 응답의 assistant response token 평균 cross-entropy
- `CE_fusion_j`: bank `j`를 사용했을 때의 assistant response token 평균 cross-entropy
- `improvement_j = CE_receiver - CE_fusion_j`

의미는 다음과 같다.

- `improvement_j > 0`: fusion이 도움 됨
- `improvement_j = 0`: 차이 없음
- `improvement_j < 0`: fusion이 해로움

`skip`의 improvement는 항상 `0`으로 둔다.

### raw action label

offline에서 최종 action label은 아래처럼 만든다.

- action set = `{skip, bank_0, bank_1, ..., bank_{K-1}}`
- score(skip) = `0`
- score(bank_j) = `improvement_j`
- `best_action = argmax(score)`

즉 label은 `skip 포함 argmax`다.

### binary target

- `best_action == skip` 이면 `0`
- `best_action != skip` 이면 `1`

### bank target

- `best_action != skip`일 때만 `best bank index`를 사용
- selection head는 positive sample에서만 학습한다


## 6. 추론 동작

query 하나에 대해 prefill을 마친 뒤, fusion 전에 routing을 1번 수행한다.

### 추론 순서

1. receiver와 sharer의 prefill KV cache를 만든다
2. router가 KV feature를 보고 `binary head`로 `skip/fuse`를 예측한다
3. `skip`이면 fusion하지 않는다
4. `fuse`이면 `selection head`가 `K개 bank 중 1개`를 고른다
5. 선택된 bank를 모든 target layer에 공통 적용한다
6. fusion은 hard top-1만 사용한다

즉 inference에서는 `binary head`가 최종 gate 역할을 하고,
`selection head`는 fuse일 때만 bank를 고른다.


## 7. 학습 전략

### 7.1 Router 학습

- receiver, sharer, projector는 freeze
- router만 학습
- 학습 데이터는 offline으로 생성
- SFT 데이터에 직접 label이 없으므로, assistant response text가 있는 데이터셋에서 CE 기반 label을 만든다

### 7.2 Projector Bank 학습

두 가지 경로를 모두 지원한다.

1. 이미 학습된 bank checkpoint를 로드해서 사용
2. bank별로 독립적으로 projector를 학습해서 bank를 생성

단, 이번 버전에서는

- 여러 bank를 한 번에 joint co-training하지 않는다
- router와 projector bank를 end-to-end로 같이 학습하지 않는다


## 8. 구현 항목

### A. Router 모듈 추가

새 파일 예시:

- `rosetta/model/router.py`

역할:

- KV feature encoder
- token pooling
- layer aggregation
- binary head
- bank selection head

### B. Wrapper 통합

수정 대상:

- `rosetta/model/wrapper.py`

변경 내용:

- entropy gate 제거
- prefill 이후 routing 실행
- binary 결과에 따라 skip/fuse 결정
- fuse이면 선택된 bank를 모든 target layer에 적용
- 기존 `pair_list`에서 첫 projector만 쓰는 동작을 bank 선택 방식으로 교체

### C. Projector Bank 로딩/저장 포맷 확장

필요 사항:

- `K`개 bank를 명시할 수 있는 config
- bank별 projector 묶음 로딩
- 기존 projector checkpoint 기반 재사용 가능

### D. Offline Label 생성 스크립트

새 script에서 해야 할 일:

1. receiver only CE 계산
2. 각 bank fusion CE 계산
3. `improvement_j` 계산
4. `best_action`, binary target, bank target 생성
5. router 입력 feature와 함께 저장

### E. Router 학습 스크립트

새 script에서 해야 할 일:

- offline dataset 로드
- router만 학습
- binary loss와 selection loss 분리
- positive sample에서만 selection loss 적용

### F. Projector Bank 생성/학습 지원

지원 방식:

- 기존 projector training을 bank별로 여러 번 돌려 bank를 만들기
- 또는 기존 checkpoint 디렉토리 여러 개를 bank로 묶기


## 9. 데이터 관점 메모

- label 계산은 assistant response token에 대해서만 수행
- 객관식 데이터뿐 아니라 일반 SFT 데이터도 사용 가능
- 정답 choice가 없어도 assistant response text만 있으면 된다
- 따라서 OpenHermes 같은 일반 SFT 데이터도 offline label 생성에 사용할 수 있다


## 10. 요약

이번 작업의 핵심은 다음 한 줄이다.

`prefill KV를 보고 query-level로 skip 또는 projector bank 하나를 고르는 C2C router를 추가한다.`

구조적으로는

- `single sharer`
- `global bank selection across all target layers`
- `binary gate + bank selector`
- `offline CE supervision`
- `hard top-1 routing`

으로 구현한다.
