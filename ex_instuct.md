wrapper.py를 수정해야함

목표는 Prefill 후 fusion하기 전에 Receiver가 답을 맞출지, sharer가 도움이 될지 예측하여 fusion 여부를 정하는 것.
그리고 fusion할 때, MoE에서 FFN 중 하나를 선택해서 연산하듯 projector를 여러 projector 중 하나를 선택해서 fusion하도록 하는 것

따라서 receiver와 sharer의 feature(KV cache)를 입력
출력은 response CE improvement

response CE improvement는 fusion을 했을 때 정답 응답 토큰들에 대한 평균 cross-entropy loss가 줄었는지를 보는 지표야.
즉 CE_receiver - CE_fusion > 0이면 fusion이 도움된 것.
객관식 정답이 없어도 assistant 응답 텍스트만 있으면 OpenHermes 같은 일반 SFT 데이터에도 쓸 수 있음.
improvement > 0: fusion이 도움 됨
improvement = 0: 차이 없음
improvement < 0: fusion이 오히려 해침

sigmoid로 확률분포로 출력하게 학습되는 probe를 만들어야 해.

probe 모델은 간단하게 linear부터 가자.



1. probe는 다른 receiver, sharer, projector는 freeze하고 학습시킬거야. wrapper에서는 추론만 하게 되겠지. projector처럼 따로 파일을 만들어서  wrapper에 넣는 방식으로 가자.
2. fusion_correct - receiver_correct를 binary하게 regression하겠다는거야. 최종적으로는 binary하게 delta>0 만 예측하면 돼.
3. routing은 쿼리 하나마다 하게 돼. 즉 샘플 하나에서 쿼리 하나가 나오고 prefill 한번 한 후 이때 fusion하기 전 routing하는거야.
4. hard top-1으로 가고, 선택할 후보 projector의 수를 학습 전에 정할 수 있어야 해.  projector 후보가 하나면 skip or fusion만 정하게 되겠지.
5. projector 내부 연산처럼럼 sequence length N을 고정 차원으로 쓰지 않고, 각 token 위치에 같은 MLP/linear를 공유 적용하자
6. SFT 학습 데이터에는 없으니까 offline으로 돌려서 만들어서 학습해야해.







