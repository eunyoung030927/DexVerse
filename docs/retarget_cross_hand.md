# 크로스 핸드 리타게팅 — 녹화된 데모를 다른 손의 데모로

공개 Shadow 데모(또는 `record_demos`로 녹화한 어떤 손의 데모)를 다른 floating hand(Allegro, LEAP, …)의 데모로
옮긴다. 학습 없이, 목표 손이 **태스크 물리 그대로** 데모의 물체 움직임을 재현하도록 SPIDER(MPPI 샘플링 최적화,
arXiv 2511.09484)로 다듬고, 태스크 자체의 성공 조건(`record_demos`와 같은 10스텝 연속 규칙)으로 판정한다.
[bench2dex_retarget](https://github.com/HyeonseokE/bench2dex_retarget)의 파이프라인(stage 1–4, SPIDER v2 설정)을
DexVerse로 옮긴 것이다.

## 실행

```bash
source /workspace/dexverse/dexverse_env.sh          # Isaac Sim + DexVerse 환경
cd <repo>
scripts/retarget/run_episodes.sh Dexverse-GraspCup-v0 \
    /workspace/dexverse/demos/demonstrations/v0/functional/Dexverse-GraspCup-v0/demos.pkl \
    floating_allegro_right 0 1 2
```

- 에피소드마다 stage 1 → 2 → 3을 차례로 돌리고(각 stage가 Isaac Sim 프로세스 하나), 끝나면 stage 4가 성공한
  에피소드만 모아 `<runs>/<task>/<demos 이름>/<target>.pkl`(record_demos 형식)을 쓴다. 이미 끝난 stage는 건너뛴다.
- 출력: `$DEXVERSE_RETARGET_RUNS`(기본 `<repo>/results/retarget`, git 제외). 로그는 그 아래 `logs/`.
- GPU: SPIDER 1024 샘플 프로세스 하나가 수 GB를 쓴다. GPU를 다른 작업과 같이 쓰면 한 번에 하나만 돌린다.
- LeRobot 데이터셋: `scripts/teleop_tools/pickle_to_lerobot.sh <target>.pkl`. 리타게팅 pickle(메타데이터 `retarget`)은
  자동으로 **상태 재생**(`record_demos --replay_demos ... --replay_states`)으로 변환된다(아래 "상태 재생" 참고).

## 파이프라인 — 에피소드 하나, 목표 손 하나

| stage | 스크립트 | 입력 | 하는 일 | 출력 |
|---|---|---|---|---|
| 1 | `scripts/retarget/stage1_reference.py` | 소스 pickle의 에피소드 (매 스텝 scene state `states`) | 기록된 상태를 소스 손 env에 그대로 써서(물리 스텝 없음) 읽는다: 물체 바디 자세, 소스 손끝 5개·손바닥, 손끝-물체 표면 접촉(2 cm), 조작 구간 | `epNNN/reference.npz` |
| 2 | `scripts/retarget/stage2_kinematic.py` | reference + 목표 손 | 목표 손 IK: 손끝 k → 소스 손끝 k (엄지..약지/소지, 없는 손가락은 버림; 접촉 중 가중치 10, 나머지 2) + 손바닥 회전(소스의 홈 대비 회전을 목표 홈 손바닥에 적용) + 직전 프레임 정규화. 손목 가상 관절 6개 + 손가락 관절을 함께 푼다. Jacobian은 시뮬레이터 USD에서 유한차분 | `epNNN/<target>/kinematic.npz` |
| 3 | `scripts/retarget/stage3_spider.py` | reference + kinematic | hold pass(물체를 데모 궤적에 고정하고 PD로 따라가 본 자세를 기준으로) 후 SPIDER: 1024개 env에 잔차 후보를 굴려 물체가 데모처럼 움직이는 계획을 고르고, 놓치면 되감아 재시도 | `epNNN/<target>/spider.json`, `spider_traj.pkl.gz` |
| 4 | `scripts/retarget/stage4_record.py` | stage 3 결과들 | 성공한 실행만 record_demos 형식의 에피소드로(성공이 10스텝 유지된 프레임에서 자름) | `<target>.pkl` |

라이브러리: `source/dexverse/dexverse/retarget/` — `hands.py`(손별 손바닥·손끝 바디, 엄지..소지 순서), `env.py`(태스크 env
N개 생성·상태 읽기/쓰기·물리 스텝·관절 목표→env action), `spider.py`(SPIDER 롤아웃 env), `geometry.py`(접촉 판정용 메시
거리, bench2dex에서 그대로), `paths.py`.

## 설계상 결정

- **태스크는 record_demos와 똑같이 만든다** (`dexverse.mimic.env_setup.build_env_cfg`: 목표 `robot_type`의 태스크 cfg,
  카메라 제거). 손 액추에이터, 물체, 테이블 물리는 벤치마크 그대로이고 바꾸지 않는다. 리타게팅은 action manager를
  거치지 않고 관절 목표를 직접 쓰고 `decimation`만큼 물리 스텝을 돈다(`env.step`과 같은 물리).
- **60 Hz 제어.** bench2dex(20 Hz)의 프레임 수 설정은 모두 ×3(같은 시간): horizon 90, knot 15, commit 15,
  gate_margin 45, gate_ramp 15, gate_look 90, settle 120. 나머지 SPIDER v2 값(샘플 1024, iters 5, sigma 0.5,
  guide_gain 300, squeeze 0.02, finger_res_cap 0.6, wrist 0.05 m / 0.5 rad, backtrack 2 commit, 재시도 3)은 그대로.
- **PD 지연 보상.** floating 손목 관절은 강성 1000·감쇠 200(시간상수 D/K = 0.2 s)이라 움직이는 목표를 0.2 s 늦게
  따라간다. 기준 명령을 `q_ref + (D/K)·dq_ref/dt`로 앞당긴다(위치 목표만 쓰므로 env action으로 그대로 표현된다).
  GraspCup ep0 → Allegro hold pass 손끝 오차가 평균 4.1 cm → 1.1 cm(잡기 전 구간 0.4 cm 이하)로 줄었다.
- **저역통과는 잔차에만.** bench2dex는 관절 목표 전체에 EMA(0.8)를 걸었는데, 그러면 움직이는 기준에 약 0.25 s 지연이
  더해진다. 여기서는 SPIDER 잔차에만 같은 시간상수(60 Hz 프레임당 0.8^(1/3))로 건다.
- **성공 판정**은 태스크의 `success` termination term을 env 0에서 매 프레임 평가하고, record_demos처럼 10프레임 연속이면
  성공이다.

## 상태 재생 — 리타게팅 pickle은 action이 아니라 state로 재생한다

SPIDER의 확정 실행은 커밋(15프레임)마다 스냅샷을 복원하며 이어 간다. 그래서 기록된 action을 초기 상태에서 끊김 없이
다시 시뮬레이션하면 접촉 캐시·솔버 warm start 차이로 결과가 달라지고, 아슬아슬한 파지에서 물체를 놓칠 수 있다(bench2dex에서
확인). 기록된 **상태**가 실제로 시뮬레이션된 결과이므로 그것을 그대로 따라가며 카메라를 렌더한다.

- `record_demos.py --replay_demos <pkl> --replay_states`: 매 스텝 뒤 기록된 scene state(`states[t]`)를 써 넣는다.
- `pickle_to_lerobot.py`는 pickle 메타데이터에 `retarget`이 있으면 이 옵션을 자동으로 붙인다.
- action 열은 그대로 들어간다(목표 손 env action 형식: 손목 6 + 손가락, `JointPositionAction` 역변환).

## 지원 범위

- floating 손, 단일(`floating_<hand>_right`)과 양손(`floating_<hand>_bimanual`): shadow, allegro, leap, sharpa, wuji
  (`hands.py`). 양손 로봇은 손마다 손바닥·손끝 5슬롯·손목 관절 6개를 갖고(오른손, 왼손 순), 손끝 슬롯 `5·손 + 손가락`으로
  소스 오른손 → 목표 오른손, 왼손 → 왼손으로 맞춘다. stage 2는 손목 12 + 손가락 관절을 한 번에 풀고, stage 3 잔차는
  손목 12 + 명령되는 손가락 관절 전부에 걸린다. action이 명령하지 않는 관절(Sharpa 양손의 head 6개)은 잔차를 받지 않는다.
- mimic 관절이 있는 손(Inspire: 손당 관절 18개 중 action 12개): `hands.py`의 `Hand.mimic`에 URDF `<mimic>` 계수를 두고,
  `RetargetEnv`가 `q = C x + d`(x = mimic이 아닌 관절)를 만든다. 계수는 stage 2가 `calibrate_mimic`으로 시뮬에서
  직접 측정한다(driver 하나씩 스윕, 자기충돌로 밀린 샘플 제외; USD의 PhysX mimic은 손가락 offset이 URDF와 다르다). stage 2는 x만 변수로 풀고(유한차분도 x 위에서), 저장되는
  `q`는 mimic까지 채운 전체 벡터다. stage 3는 물리 스텝이라 PhysX mimic 제약이 그대로 잡고, 잔차는 driver에만 걸린다.
- 물체: 동적 rigid object와 로봇이 아닌 articulation(링크 단위). 테이블·`success_marker`는 접촉 대상에서 뺀다.

## 결과

| 태스크 · 에피소드 | 목표 손 | stage 2 손끝 오차 | 결과 | 컵 오차 평균 | 재시도 | 시간 (RTX 3090) | 상태 재생 / action 재생 |
|---|---|---|---|---|---|---|---|
| GraspCup v0 · ep 0 | floating_allegro_right | 0.17 cm | 성공 (280프레임) | 2.3 cm | 4 | 35.8분 | 성공 / 재현 안 됨 |

영상 비교: `scripts/retarget/render_states.py --pickle <pkl> --episode N --out <mp4>`(기록된 상태를 그대로 렌더).
