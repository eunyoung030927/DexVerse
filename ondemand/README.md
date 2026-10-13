# ondemand — DexVerse 손 간 retargeting SLURM 잡 (pro6000, OpenOnDemand)

DexVerse v1 공개 데모(Shadow)를 다른 손 5종으로 SPIDER retargeting해서 trajectory pickle로 모으고 HF에 올리는 클러스터 잡.
Bench2Dex의 `ondemand/bench2dex/`(HyeonseokE/bench2dex_retarget)와 같은 구조다. 이 폴더는 잡·컨테이너·셋업·다운로드만 담고,
retargeting 코드는 `scripts/retarget/`(stage 1–4, 큐, 업로드)와 `source/dexverse/dexverse/retarget/`에 있다 —
[`docs/retarget_cross_hand.md`](../docs/retarget_cross_hand.md). (2026-10-12 기준)

## 목표

| | |
|---|---|
| 소스 | `dexverse/DexVerse_release` `demonstrations/v1/<category>/<task>/demos.pkl` (Shadow, 태스크당 50개) |
| 태스크 (9) | 한 손(`floating_shadow_right`): GraspBleach, GraspPan, GraspKettle, FunctionalPourMug, FunctionalHammerStrike<br>양손(`floating_shadow_bimanual`): BimanualLiftTray, BimanualLiftCarton, SlideUtilityKnife, OpenFlatFolder |
| 목표 손 (5) | Allegro, Inspire, LEAP, Sharpa, Wuji — 한 손 태스크는 `floating_<hand>_right`, 양손 태스크는 `floating_<hand>_bimanual` |
| 규모 | 9 태스크 × 5 손 = 45 조합, 조합당 50 에피소드 |
| 출력 | 동료(Eun) 레코딩 형식인 `record_demos` trajectory pickle. LeRobot은 렌더링 되는 다른 서버에서 `scripts/teleop_tools/pickle_to_lerobot.sh`(상태 재생)로 만든다 |

v1 환경 재현 확인(2026-10-12, 9 태스크 × 5 에피소드): 기록된 상태 재생 45/45 성공, action 재생 3/45.
retargeting은 상태를 입력으로 쓰므로 그대로 진행할 수 있다(`/workspace/dexverse/results/v1_replay_check/`).

## 해야 할 일

| # | 할 일 | 상태 |
|---|---|---|
| 1 | 클러스터 파이프라인 (`ondemand/`, `scripts/retarget/run_queue.py`·`run_target.py`·`upload_task.py`) | 완료. 로컬(GraspBleach-v1 ep0, SPIDER 90프레임 축소)에서 큐 → stage 1–3 → 상태 기록 → stage 4 pickle → 업로드 판정(dry run) → 그 pickle의 상태 재생까지 확인. 클러스터에서는 아직 안 돌림 |
| 2 | **Sharpa** 지원: `hands.py`에 손바닥·손끝 바디, `right_*` 손목 관절 | 추가함. stage 1–3 동작 확인(stage 2 손끝 오차 평균 0.05 cm), 전체 길이 성공은 미확인 |
| 3 | **Inspire** 지원: 관절 18개 중 6개가 mimic(따라 움직이는 관절, action은 12). stage 2 IK를 action 12개 기준으로 풀고 mimic 관절은 연동식으로 채움 (Bench2Dex RH56DFX `coupling` 방식) | 구현함 (2026-10-13). `hands.py`에 Inspire 오른손·양손 스펙 + URDF `<mimic>` 계수(driver→mimic 비율·오프셋), stage 2는 mimic을 뺀 관절(손목 6 + driver 6 = action 12)만 변수로 풀고 mimic은 `q = C x + d`로 채움. 계수는 stage 2가 시뮬에서 직접 측정(`RetargetEnv.calibrate_mimic`: USD의 PhysX mimic은 손가락 offset이 URDF의 -0.045가 아니라 0). GraspBleach ep0 stage 1–3 동작 확인(stage 2 손끝 오차 0.94 cm, hold pass 3.3 cm) |
| 4 | **양손** 지원 (태스크 4개): `hands.py`에 `floating_<hand>_bimanual` 스펙(손별 손목 6관절·손끝 10개), stage 1–3를 손 개수에 대해 일반화 | 구현함 (2026-10-12). Allegro·LEAP·Sharpa·Wuji 양손; 로컬에서 LiftCarton ep15 stage 1–3 배관 확인 |
| 4b | **LEAP 폭주 수정** (2026-10-13): stage 2 IK는 자기충돌을 모르므로 겹친 자세로 텔레포트하면 PhysX depenetration이 손을 튕겨냄(hold pass 118 cm, 손목 50 rad). stage 3 초기 자세를 홈에서 기준 0프레임으로 PD 접근해 안착한 자세로 바꿈(`SpiderEnv.initial_state`; 물리 설정 변경 없음) | 수정함. LEAP hold pass 118 → 13.9 cm(자기충돌·물체 접촉으로 손가락 2.5 rad 편차는 남음), Allegro 6.7 → 5.7 cm. `dd78096`으로 도는 잡의 LEAP 항목은 전부 실패 → 수정본으로 재제출 권장 |
| 4c | **stage 2 특이행렬 수정** (2026-10-13, 클러스터 잡 1649에서 Wuji/Sharpa `stage2_failed`로 발견): 관절이 한계값에 붙으면 유한차분 프로브가 한계로 클램프돼 분모 ~0 → `torch.linalg.solve: singular`. 프로브를 한계 쪽에서는 -eps로 돌리고, 정규방정식은 float64로 풂 | 수정함. GraspBleach ep1 Wuji(클러스터 실패 건) 로컬 0.51 cm, Sharpa 0.06 cm |
| 5 | 각 (태스크, 손)을 로컬에서 에피소드 1개로 끝까지 검증한 뒤 클러스터 제출 | 한 손 태스크 × Allegro/LEAP/Sharpa/Wuji부터 (Wuji는 손목 관절 이름 수정 후 stage 1–3 동작) |
| 6 | `env_check`: 클러스터에서 TASKS 하나, EPISODES=1로 먼저 돌려 셋업·다운로드·stage 1–3 확인 | 제출 전 |

지원하지 않는 손은 큐가 `skip ... not supported` 한 줄을 남기고 건너뛴다. 2026-10-13부터 5개 손 모두 지원(9 × 5 = 45 조합).
`dd78096`으로 시작한 잡은 Inspire를 건너뛰고, continuation job(포크를 다시 pull)부터 Inspire 항목이 큐에 들어간다.

주의: DexVerse README에 따르면 `BimanualLiftTray-v1`에서 Allegro/LEAP 양손은 명령한 손목·손가락 응답이 부족했다(v0는 통과).

## 붙여넣는 파일 (OOD Job Composer → New Job → 내용 붙여넣기 → Submit)

| 파일 | 무엇 |
|---|---|
| `main_job.sbatch` | **실제 작업.** pro6000 2장을 처음부터 잡고 한 잡에서 셋업 → 다운로드 → retargeting → HF 업로드 판정. 태스크 하나당 잡 하나, `TASK_QUEUE` 순서로 다음 태스크 잡을 자동으로 이어 붙임 |
| `upload_job.sbatch` | 끝난 태스크를 지금 업로드(기본 부분 업로드). 나중에 빠진 에피소드만 채워 올리면 HF의 pickle과 합쳐 50/50으로 갱신 |

나머지 파일은 잡이 pull한 fork 체크아웃(`$HOME/dexverse/DexVerse`, 브랜치 `retarget/shadow-cross-hand`)에서 읽는다.
그래서 `ondemand/*.sh`, `scripts/retarget/`, `source/`를 고치면 push만 하면 되고, 붙여넣는 sbatch 자체를 고쳤을 때만 다시 붙여넣는다.

## `main_job.sbatch` 흐름 (Bench2Dex와 같음, 바뀐 곳은 굵게)

1. preflight — `~/.hf_token` 확인
2. git sync — **fork `eunyoung030927/DexVerse`의 `retarget/shadow-cross-hand`** 를 clone 또는 fast-forward. 다음 태스크 잡(`afterany`)을 이 시점에 대기열에 넣음
3. setup (`setup.sh`, 최초 1회) — Isaac Sim 5.1 SIF(**Bench2Dex가 만든 `$HOME/b2d/sif`가 있으면 재사용**), Isaac Lab v2.3.2,
   **DexVerse 패키지와 의존성, 손 6종 에셋 + core_assets + ManiTwin 물체**(렌더링하지 않으므로 HDRI·long-horizon 제외)
4. fetch (`fetch_data.py`) — **태스크의 v1 데모 pickle**(`dexverse/DexVerse_release`)
5. retarget (`retarget_worker.sbatch`) — CUDA MPS, GPU당 `PER_GPU`(16) 세션, VRAM `MIN_FREE_GB` 남을 때만 새 세션(2 GPU에서 실측 34개 동시). (에피소드, 목표 손) 큐를
   **`scripts/retarget/run_queue.py` → `run_target.py`** 로 처리: 에피소드마다 stage 1 한 번, 손마다 stage 2(기구학 IK) → stage 3(SPIDER v2, 60 Hz).
   판정은 **태스크 자신의 `success` 조건 10스텝 유지**(record_demos 규칙). 렌더링 없음
6. upload (`finalize.sbatch` → **`scripts/retarget/upload_task.py`**) — 손별로 50개 모두 성공했을 때만,
   **stage 4로 성공한 실행을 pickle 하나로 모아** 데이터셋 카드와 함께 올림

- 시도: (에피소드, 손)당 최대 `MAX_ATTEMPTS`(3)회. 시도 K: 시드 K, 샘플 1024·(1+K//2), iters 5+K (Bench2Dex와 같음).
  성공한 시도는 `spider.json` / `spider_traj.pkl.gz` 심볼릭 링크로 표시
- 태스크는 1번 → 2번 → … 순서로 하나씩 돈다: 각 잡이 다음 태스크 잡을 `afterany`로 걸어 두고, 다른 dvr 잡이 돌고 있으면 그 뒤로 재대기한다
- 다시 제출하면 끝난 (에피소드, 손)은 건너뜀

## Bench2Dex에서 바뀐 부분

| | Bench2Dex | DexVerse |
|---|---|---|
| 레포 · 작업 폴더 | `HyeonseokE/bench2dex_retarget` · `$HOME/b2d` | `eunyoung030927/DexVerse` (`retarget/shadow-cross-hand`) · `$HOME/dexverse` (컨테이너 `/workspace/dexverse`) |
| 태스크 ID | `06`, `79` … | `Dexverse-GraspBleach-v1` … |
| 목표 손 | 태스크별 목록 (`TASK_TARGETS`) | 손 이름 목록 `HANDS`, 소스 robot_type의 손만 바꿈 (`_right` / `_bimanual` 유지) |
| 소스 데이터 | `Bench2Dex/teleopdata` HDF5 | `dexverse/DexVerse_release` v1 `demos.pkl` |
| 성공 판정 | MetricTracker | 태스크의 `success` termination, 10스텝 연속 |
| 기록 | 에피소드마다 Bench2Dex HDF5 | 성공한 SPIDER 실행 → (업로드 때) stage 4가 pickle 하나로 모음, `action_layout`은 목표 손 것 |
| HF 레포 | `<ns>/b2d-<scene>-<hand>-retargeting` | `<ns>/dexverse-<task>-<hand>-retargeting` (예: `dexverse-graspbleach-v1-allegro-retargeting`) |
| HF 내용 | `dataset/<scene>/<hand>/origin-generalization/episode_NNNNNN.hdf5` | `demonstrations/v1/<category>/<task>/demos.pkl` (공개 릴리스와 같은 경로) + `README.md` + `failures.json` |
| 부분 업로드 | 에피소드 파일 단위로 합침 | HF의 `demos.pkl`을 받아 소스 에피소드 번호(`retarget_source.episode`)로 합침 |
| 상태 파일 | `~/b2d/b2dr_status_task<T>.md` | `~/dexverse/dvr_status_<task>.md` |

## 조절 (sbatch 상단 기본값 수정, 또는 `--export=ALL,KNOB=값`)

| knob | 기본 | 의미 |
|---|---|---|
| `TASK_QUEUE` | 9개 태스크 | 이 순서로 태스크당 잡 하나씩 |
| `TASKS` | 큐의 첫 태스크 | `TASKS=Dexverse-GraspPan-v1`만 주면 그 태스크 하나 (체인 없음) |
| `HANDS` | `allegro inspire leap sharpa wuji` | 목표 손 |
| `EPISODES` | `50` | 소스 에피소드 0..N-1 |
| `MAX_ATTEMPTS` | `3` | (에피소드, 손)당 SPIDER 시도 횟수 |
| `PER_GPU` · `MIN_FREE_GB` · `SESSION_GB` · `MPS_ON` | `16` · `10` · `6` · `1` | GPU당 세션 상한 · 새 세션을 띄울 최소 여유 VRAM · 세션당 VRAM 가정 · MPS |
| `SPIDER_ARGS` | 없음 (v2) | stage 3 추가 인자 |
| `UPLOAD` · `HF_NAMESPACE` | `1` · 토큰 사용자 | 업로드 |

## 진행 확인 (클러스터 셸 없음 → 잡이 스스로 보고)

OOD Files 앱에서 `~/dexverse/dvr_status_<task>.md`(5분마다 갱신): (태스크, 손)별 성공·실패·진행 수, 첫 시도/재시도 성공,
큐의 최근 줄, 오류, GPU 메모리·세션, MPS, `/tmp` 여유. 같은 요약이 잡 로그에 `[monitor ...]`로 남는다.

## 출력 위치 (클러스터)

| 무엇 | 경로 |
|---|---|
| (에피소드, 손)별 결과: `status.json`, `run.log`, `spider_aK.json`·`_traj.pkl.gz`, `spider.*` | `~/dexverse/runs/retarget/<task>/demos/epNNN/<target>/` |
| stage 1 reference | `~/dexverse/runs/retarget/<task>/demos/epNNN/reference.npz` |
| 세션 로그 | `~/dexverse/runs/retarget/logs/` |
| 진행 표 (레포 `experiments/`와 같은 구조: STATUS.md, `<task>/<target>/STATUS.md`·`results.csv`·`config.yaml`) | `~/dexverse/runs/report/` — 결과를 기록할 때 레포 `experiments/`로 복사 |
| 잡 로그 | 제출 폴더의 `slurm-dvr-main-<job>.out`, `OUTPUTS.md` |

## 파일

| 파일 | 역할 |
|---|---|
| `env.sh` | 경로(`$HOME/dexverse` → 컨테이너 `/workspace/dexverse`), SIF 선택, apptainer 래퍼(`ISAAC`), 토큰, `sync_repo` |
| `setup.sh` | SIF, Isaac Lab, python 의존성, DexVerse 에셋 (1회, 스탬프) |
| `fetch_data.py` | 태스크의 v1 데모 pickle 다운로드 (HF 429는 기다렸다 이어 받음) |
| `retarget_worker.sbatch` | 2 GPU 워커: MPS, GPU id 확인, `run_queue.py`, 끝에 손별 업로드 가능 여부(dry run) 출력 |
| `monitor.sh` · `outputs_note.sh` | 상태 파일 · 잡 폴더의 `OUTPUTS.md` |
| `finalize.sbatch` | 손별 업로드 판정 (50/50일 때만 pickle + 카드) |
| `upload_job.sbatch` | (붙여넣기용) 부분 업로드 |

**실측 (잡 1651, 2026-10-13):** p1의 pro6000 2장, 세션 34개 동시(GPU 사용률 100%, VRAM 66/98 GB), GraspBleach 한 손 항목당
Allegro 28분·Inspire 46분(1차 성공 기준). 재시도 포함 평균 ~1시간/항목으로 보면 처리량 ≈ 34항목/시간 → 한 손 1,250항목 ≈ 1.5일,
양손 1,000항목(프레임 1.5배·관절 2배) ≈ 3.5~4일, 전체 5~6일. 양손 실측이 쌓이면 다시 잡는다.

**노드 사양 (p1, sysinfo 잡 1534 — Bench2Dex `experiments/ENVIRONMENT.md`):** pro6000 96 GB **4장**, EPYC 9135 ×2 = 64스레드,
RAM 500000M, `/tmp` 7 TB(99% 사용). 현재 잡(`gres 2 / cpus 48 / mem 320G`)은 사양 안. 노드 전체를 쓰려면 `main_job.sbatch` 상단
`--gres=gpu:pro6000:4 --cpus-per-task=64 --mem=440G`, `PER_GPU=11`(세션당 ~1.4코어 → 64스레드에 ~45세션). CPU가 먼저 한계라
2장 대비 처리량은 1.3~2배 사이(세션당 GPU 몫이 커져 빨라지는 만큼). 워커는 잡이 받은 GPU 수를 그대로 쓴다.
