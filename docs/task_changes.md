# 태스크 변경 기록

저자 배포 태스크(v0/v1)를 이 저장소에서 바꾼 내역. 바뀐 태스크로 이전에 모은 데이터는 장면이 달라 재현되지 않으므로,
데이터를 모을 때는 pickle의 `provenance.git_commit`이 아래 변경 이후 커밋인지 확인한다.

| 날짜 | 태스크 | 커밋 |
|---|---|---|
| 2026-10-28 | Dexverse-GraspTwoItems-v0 | b49a1a5 |
| 2026-10-29 | 모든 태스크 — record_demos 에피소드별 리셋 시드 (`--episode_seed`) | 35d584c |

## 2026-10-28 · Dexverse-GraspTwoItems-v0 — 정육면체 2개만 사용

- **변경 전**: 물체 5종(구 지름 5.6 cm, 원기둥 지름 4.4 × 높이 6 cm, 삼각기둥 한 변 6 × 높이 5.5 cm,
  정육면체 4.5 cm, 직육면체 6.5 × 3.5 × 4.5 cm, 각 0.06 kg) 중 매 리셋 2개를 무작위로 골라 테이블에 놓고,
  나머지 3개는 테이블 아래(z = −10 m)로 치움.
- **변경 후**: 4.5 cm 정육면체 2개(빨강 `cube`, 파랑 `cube_2`, 각 0.06 kg)를 매 에피소드 모두 놓음.
- **그대로인 것**: 배치(테이블 중앙 ±6 cm, 물체 간 반경 합 + 1.2 cm 이상, 방향 무작위),
  성공 조건(두 물체 모두 테이블 면 + 0.20 m 이상), 실패 조건(테이블 범위 이탈, 20 s 시간 초과).
- **이유**: 텔레옵 수집에서 구·원기둥·삼각기둥·직육면체는 한 손으로 두 개를 함께 집기가 너무 어려워
  성공 데모를 모으기 힘들었음. 잡기 쉬운 정육면체로 통일해 수집 성공률을 높임.
- **데이터 영향**: 이전 5종 장면으로 모은 GraspTwoItems-v0 pickle은 장면 물체 이름(sphere, cylinder 등)이 달라
  replay·변환이 맞지 않음 → 버림 (2026-10-28 결정).
- **코드**: `source/dexverse/dexverse/tasks/config/grasping/grasp_two_items_cfg.py`
  (`PRIMITIVE_OBJECT_SPECS`, `GraspTwoItemsSceneCfg`).

## 2026-10-29 · record_demos — 에피소드 번호로 리셋 장면을 고정 (`--episode_seed`, 기본 켜짐)

- **배경**: 손 6종에 대해 같은 태스크의 데모를 모을 때 "에피소드 k의 물체 배치가 모든 손에서 같아야" 손 간 비교
  (cross-hand) 데이터가 된다. 기존에는 세션 시작 때 `cfg.seed`(기본 42)로 한 번만 시드를 걸고 리셋마다 같은
  난수열을 이어 썼다. 난수 소비량이 실패 횟수·에피소드 길이·손(관절 수, 재질 랜덤화 대상 바디 수)에 따라
  달라서, 시드가 같아도 **k번째 에피소드의 장면은 손·세션마다 달랐다** (GraspPan-v0, 5손, 시드 42: 첫 리셋만 같고
  두 번째부터 다름). 또 RESET 버튼으로 다시 시작하면 **다음 난수의 새 배치**가 나왔다.
- **변경**: 매 리셋 직전에 `(기본 시드 × 1,000,003 + 다음에 저장될 에피소드 번호)`로 모든 RNG(torch·numpy·python·
  warp·replicator)를 다시 시드하고, 리셋 이벤트 항목(`events.reset_*`) 각각에도 `에피소드 시드 + 항목 이름 해시`로
  독립 시드를 건다(한 항목의 난수 소비량이 손에 따라 달라도 다른 항목에 영향 없음). 결과:
  - 에피소드 k의 물체·목표·보조물 배치가 **손·세션 무관하게 동일** (기본 시드가 같을 때).
  - **RESET·실패(물체 이탈 등)로 다시 시작하면 같은 배치** — 에피소드 번호가 올라가지 않으므로.
  - 성공 저장 뒤의 리셋은 다음 배치. 세션 첫 리셋 = 에피소드 0.
  - 검증(2026-10-29, a4, GraspPan-v0, allegro·leap·shadow 별도 프로세스): ep0/ep1/ep2 와 각 재시도의 물체·목표·스토브
    위치가 3손에서 모두 동일.
- **켜기/끄기**: `record_demos.py --episode_seed`(기본) / `--no_episode_seed`(이전 방식: 세션당 한 난수열).
  `dexverse_session.sh record … --no_episode_seed` 로 그대로 전달된다. `--replay_demos`에서는 자동 off(초기 상태를
  pickle에서 복원하므로). 어느 쪽으로 모았는지는 pickle `provenance.env["DEXVERSE_EPISODE_SEED"]` = `"1"`/`"0"` 과
  콘솔의 `[episode seed] episode k: seed …` 로 남는다.
- **같은 배치를 다른 손에서 재현하려면**: 같은 태스크, 같은 `--seed`(기본 42), `--episode_seed` 켜고, 같은 에피소드
  번호. 로봇 손의 시작 포즈는 손별 고정값이라 손끼리는 다르다(사람은 START 정렬로 맞춤).
- **데이터 영향**: 이 변경 전에 모은 pickle(2026-10-29 이전: GraspPan-v0 5손 × 10, GraspKettle-v0 wuji·sharpa 50,
  GraspTwoItems-v0 shadow 50 등)은 손 간 장면이 정렬되어 있지 않다. 손 간 정렬이 필요한 용도면 재수집 대상.
  단일 손 학습 데이터로는 그대로 쓸 수 있다.
- **코드**: `scripts/record_demos.py`(`--episode_seed`, `handle_reset`, `_next_episode_index`),
  `source/dexverse/dexverse/data_collection/episode_seed.py`(`seed_for_episode`, `install_per_term_seeding`).
