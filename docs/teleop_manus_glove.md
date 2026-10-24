# DexVerse 텔레옵 — Manus 글러브 + Quest 3 컨트롤러

손가락은 **Manus Metagloves Pro**, 손목 6D pose는 **손등에 단 Quest 3 컨트롤러**로 받아 DexVerse floating 손을
VR(CloudXR 6)로 텔레옵한다. VR 화면·헤드셋 메뉴(START/STOP/RESET)는 기존 Quest 텔레옵
([teleop_quest_cloudxr6.md](teleop_quest_cloudxr6.md))과 같고, 입력 장치만 `--teleop_device manus`로 바뀐다.

## 구성

```
[Manus 글러브] ─BT─ [동글] ─USB─ 컨테이너: robo-viz/manus/bridge.py (Manus SDK Integrated)
                                      │ TCP 127.0.0.1:47000, 25관절 + 손목 쿼터니언, ~100 Hz
[Quest 3 + 손등 컨트롤러] ─Tailscale─ CloudXR 6 런타임 ─ Isaac Sim (XR)
                                      │
                    ManusDevice (devices/manus/manus_device.py)
                      ├ 손가락: 글러브 손 모양 (Manus 손목 노드 기준)
                      └ 손목:   컨트롤러 pose + 피벗 오프셋 + START 때 잡은 회전 오프셋
                                      │ OpenXR 손 관절 26개 형식 (Quest 손 추적과 같은 형식)
                    Bench2DexFingerRetargeter (devices/manus/bench2dex_retargeter.py)
                      ├ 손목: SimpleRelativeRetargeter (START 기준 변화량, clutch)
                      └ 손가락: Bench2Dex RetargetBridge (Manus용으로 튜닝된 DexPilot)
```

- 손가락 리타게팅은 Bench2Dex 텔레옵 코드(`bench2dex_teleop/teleop`, 커밋 511cbf3)의 `retarget_bridge.py`와
  `hand_cfgs/`를 수정 없이 `devices/manus/bench2dex/`에 복사해 쓴다. URDF는 Bench2Dex URDF-Zoo(`$URDF_ZOO_DIR`).
- 손목은 relative 방식이다: 로봇 손은 START 순간의 자세에서 출발해 사람 손목이 움직인 만큼만 움직인다.
  START 때 사람 손을 로봇 손에 겹치게 맞추므로(아래) 이후 계속 겹쳐서 따라온다.

## 1회 준비

| 항목 | 위치 | 비고 |
|---|---|---|
| Manus Linux SDK 3.2.1 | `/workspace/manus_sdk` | Python 패키지는 미리 빌드돼 있음. `cffi` 필요 |
| 글러브 설정·캘리브레이션 | `/root/.config/Manus → /workspace/manus_config` | 페어링·캘리브레이션은 Windows MANUS Core에서 |
| Manus 브리지 | `/workspace/robo-viz/manus/bridge.py` | 손목 쿼터니언(`wq`)을 같이 보내는 버전이어야 함 |
| Bench2Dex URDF-Zoo | `/workspace/assets/bench2dex/URDF-Zoo` | `URDF_ZOO_DIR` (`/workspace/dexverse_env.sh`에서 설정) |
| CloudXR 런타임 환경 | `/workspace/envs/bench2dex/isaacteleop` | `ISAACTELEOP_CONDA_ENV` |
| Tailscale | 서버 호스트 | Quest와 같은 tailnet. 서버 IP는 `tailscale ip -4` |

동글은 한 프로그램만 쓸 수 있다. 브리지 하나만 SDK를 띄우고 나머지는 TCP로 받는다.

## 실행 (터미널 3개, 모두 컨테이너 안)

```bash
docker exec -it dexverse bash
source /workspace/dexverse_env.sh
```

```bash
# 터미널 1 — Manus 브리지 (5초마다 gloves=[('0x...', 'right')] 이면 정상)
cd /workspace/robo-viz/manus && MANUS_SDK_PY=/workspace/manus_sdk/Python python -u bridge.py

# 터미널 2 — CloudXR 런타임 (영상이 Tailscale로 가도록 서버 Tailscale IP 지정)
CXR_ENDPOINT_IP=<서버 Tailscale IP> /workspace/DexVerse/scripts/teleop_tools/start_cloudxr_runtime.sh

# 터미널 3 — 텔레옵
cd /workspace/DexVerse
DEXVERSE_MANUS_WRIST=controller \
DEXVERSE_MANUS_CTRL_OFFSET=/workspace/datasets/dexverse/calibration/controller_wrist_offset_both.json \
scripts/teleop_tools/run_teleop.sh teleop_agent \
  --task Dexverse-PushT-v1 --robot_type floating_shadow_right --teleop_device manus
```

- 데모 녹화는 `teleop_agent` 대신 `record_demos` (`--num_demos N` 등은 [teleop_quest_cloudxr6.md](teleop_quest_cloudxr6.md)).
- 단일손 태스크는 모두 오른손 기준(`floating_*_right`)이다. 왼손은 `--robot_type floating_*_left`로 쓸 수 있지만
  장면은 오른손에 맞춰져 있다.
- 글러브 값만 보려면 `python /workspace/robo-viz/manus/client.py`.

### Quest 3

1. **컨트롤러 모드**: 설정 → 움직임 추적 → 손 및 신체 추적에서 손 추적을 끈다 (마운트에 단 컨트롤러를
   "내려놓음"으로 보고 손 추적으로 바뀌는 것을 막기 위해).
2. 브라우저에서 `https://<서버 Tailscale IP>:48322` 인증서 수락 → `https://nvidia.github.io/IsaacTeleop/client`
   → Server IP, Port `48322` → Connect → Enter VR. 텔레옵을 다시 띄울 때마다 Disconnect → Connect.
3. 로봇은 **START 전에는 움직이지 않는다**. 빨간 점(내 손)만 움직인다.

### START (손 맞추기)

1. 오른손을 실제 책상에서 약 20 cm 띄운다 (시뮬레이션: 테이블 윗면 0.6 m, 로봇 손목 홈 0.8 m).
2. 실제 손을 VR 속 **로봇 손과 같은 방향**으로 둔다 (빨간 점은 START 전엔 컨트롤러 방향이라 무시).
3. 헤드셋 메뉴 **START** → 로그:
   - `[manus] right controller->hand rotation calibrated from robot hand orientation (... deg)` — 회전 오프셋
   - `[manus] recentered: XR anchor moved by [...] m` — 시점을 옮겨 내 손을 로봇 손 위치에 겹침
   - `Calibrated wrist pose for HAND_RIGHT` — 이후 로봇이 따라옴
4. 어긋나면 **STOP → 손 다시 맞추기 → START** (clutch).

## 피벗 캘리브레이션 (컨트롤러 → 손목 회전 중심 오프셋)

마운트를 새로 끼우거나 위치가 바뀌면 다시 한다. 텔레옵 대신 실행:

```bash
cd /workspace/DexVerse
scripts/teleop_tools/run_teleop.sh scripts/teleop_tools/record_controller_poses.py \
  --out /workspace/datasets/dexverse/calibration/pivot_right_$(date +%Y%m%d-%H%M%S).npz
```

1. Quest 접속 (컨트롤러 모드).
2. 손목 바깥쪽을 책상 모서리에 대고 고정한다.
3. **반대 손으로 그 컨트롤러의 트리거(검지 버튼)를 꾹 누른 채** 20~30초 동안 위아래·좌우로 꺾고 **비튼다**.
   트리거를 누르고 있는 동안만 기록된다.
4. 그 컨트롤러의 **A(오른쪽) / X(왼쪽)** 버튼 → 저장, 결과 출력, 프로그램 종료.
5. 텔레옵용 JSON 저장: `python scripts/teleop_tools/pivot_calibration.py <npz> --out <json>`
   (좌·우를 한 파일에 쓰려면 두 결과의 `left` / `right` 항목을 합친다. 현재: `controller_wrist_offset_both.json`)

판정: RMS 5 mm 이하, 가장 좁은 회전 범위 30° 이상이면 좋음. 경고가 나와도 쓸 수는 있다(손목 회전 시 그만큼
로봇 손이 흔들림). 원리: 손목을 고정하면 컨트롤러는 손목 중심의 구 위를 움직인다 → `t_i + R_i r = p`를
최소제곱으로 풀어 `r`(컨트롤러 기준 손목 위치)을 구한다. 회전 오프셋은 이 방법으로 알 수 없어 START 때 잡는다.

## 옵션 (환경변수)

| 변수 | 값 | 기본 |
|---|---|---|
| `DEXVERSE_MANUS_WRIST` | `controller` / `handtracking`(Quest 손 추적, 글러브 착용 시 부정확) / `fixed`(손목 고정) | `fixed` |
| `DEXVERSE_MANUS_CTRL_OFFSET` | 피벗 결과 JSON | 없음(오프셋 0) |
| `DEXVERSE_MANUS_FINGERS` | `bench2dex` / `dexverse`(Quest 손 추적용 DexVerse YAML) | `bench2dex` |
| `DEXVERSE_MANUS_AUTO_START` | `1`이면 손 데이터가 들어오자마자 시작 | `0`(헤드셋 START) |

손가락 보정: `source/dexverse/dexverse/devices/manus/bench2dex/hand_cfgs/<손>.yml`의 `hand_scale_xyz`,
`hand_offset_xyz`, `joint_limits_override`. 실행 중에 고쳐도 약 2초 안에 반영된다.
지원 손: shadow, allegro, leap, sharpa, wuji, inspire(=Bench2Dex `rh56dfx`).

## 시각화 (`--enable_debug_vis`가 켜진 v1 태스크)

| 표시 | 의미 |
|---|---|
| 빨간 점 | 장치가 넘긴 손 관절 26개 (월드 좌표) = 글러브 손 모양을 손목 pose에 놓은 것 |
| 파란 점 | DexVerse 기본 리타게터용으로 손목 기준·고정 방향으로 정규화한 손 모양 (Bench2Dex 입력과는 축이 조금 다름) |
| 손목 좌표축 | 장치가 넘긴 손목 pose |
| 로봇 손 | 리타게팅 결과로 움직인 실제 로봇 (relative: START 기준 변화량) |

## 헤드리스 점검 (VR 없이)

```bash
python /workspace/robo-viz/manus/bridge.py --fake --port 47001 &   # 글러브 없이: 합성 손
python scripts/teleop_tools/manus_finger_check.py --task Dexverse-PushT-v1 \
  --robot_type floating_shadow_right --port 47001 --steps 300 --headless --enable_cameras
```
손목 고정(`fixed`)으로 손가락 리타게팅만 확인하고, `<out>/finger_joints.csv`와 카메라 영상을 남긴다.

## 트러블슈팅

| 증상 | 확인 |
|---|---|
| 브리지 `gloves=[]` | 글러브 전원·절전. Windows Core에서 페어링했는지 |
| `no controller registered` (CloudXR 로그) | Quest가 손 추적 모드로 접속함 → 컨트롤러 모드로 다시 Connect |
| `controller tracked at [anchor 좌표]` 뒤 로봇이 엉뚱한 곳 | Quest 접속 전 빈 값. 접속 후 STOP → START |
| 빨간 점이 컨트롤러 방향을 봄 | START 전 정상. START 때 회전 오프셋이 잡힘 |
| 로봇 손 방향이 내 손과 일정 각도 어긋남 | START 때 손 방향을 덜 맞춘 것 → STOP → 맞추기 → START |
| 손목만 돌렸는데 로봇 손이 원을 그림 | 피벗 오프셋 없음/부정확 → 피벗 캘리브레이션 |
| 시뮬레이션 책상이 낮음 | START 때 손을 책상에서 약 20 cm 띄우기 |
| 피벗 기록 다음 실행에서 `left/right=none` | 이전 프로그램이 남아 VR 세션을 잡음 → 종료 후 Disconnect → Connect |

## 변경 파일

- `source/dexverse/dexverse/devices/manus/` — `manus_device.py`, `bench2dex_retargeter.py`, `bench2dex/`(복사본)
- `scripts/teleop_agent.py`, `scripts/record_demos.py` — `--teleop_device manus` 연결, 위 환경변수
- `scripts/teleop_tools/run_teleop.sh` — Isaac Sim 번들 python이 없으면 활성 conda 환경 python 사용
- `scripts/teleop_tools/record_controller_poses.py`, `pivot_calibration.py` — 피벗 캘리브레이션
- `scripts/teleop_tools/manus_finger_check.py` — 헤드리스 점검
- 저장소 밖: `/workspace/robo-viz/manus/{sdk_reader,bridge}.py`에 손목 쿼터니언(`wq`) 송신 추가
