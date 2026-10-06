# DexVerse VR 텔레옵 — Quest 3 + CloudXR 6 (isaacteleop)

upstream README의 텔레옵 절차(Apple Vision Pro + CloudXR 런타임 docker compose)와 달리, 이 문서는
**Meta Quest 3 브라우저(WebXR) + CloudXR 6.1 런타임(isaacteleop pip 번들)** 구성으로 DexVerse 태스크를
텔레옵하는 방법이다. 서버 쪽 구성은 vls 컨테이너에서 검증한 RH5DG2 텔레옵과 같다.

## 구성

```
[Quest 3 브라우저] ──Tailscale──▶ [work1 호스트, --network host]
                                   └ dexverse 컨테이너 (isaaclab-vnc:2.3.2-conda)
                                       ├ CloudXR 6.1 런타임   (conda env isaacteleop, WSS 48322)
                                       └ Isaac Sim + DexVerse (kit python, XR headless)
```

- 손 리타게팅은 DexVerse 내장 `SimpleRelativeRetargeter`/`SimpleAbsoluteRetargeter` + 손별
  `robot_agents/<hand>/retarget/{side}_{dexpilot,vector}.yml` 을 그대로 쓴다. `--robot_type`으로 손을 바꾼다
  (`floating_{shadow,allegro,inspire,sharpa,wuji}_{right,left,bimanual}`, `floating_leap_{right,bimanual}`).
- CloudXR 런타임(포트 48322)은 **호스트당 하나**만 뜰 수 있다. vls 컨테이너에서 런타임이 떠 있으면 먼저 내린다.

## 1회 설치 (컨테이너를 새로 만든 경우)

```bash
docker exec -it dexverse bash
# DexVerse 패키지 (NAS 레포, 브랜치 teleop/quest-cloudxr6)
/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e /workspace/dexverse/DexVerse/source/dexverse
# isaacteleop + CloudXR 런타임 (conda-forge 채널만 사용 → Anaconda ToS 불필요)
source /opt/conda/etc/profile.d/conda.sh && unset PYTHONPATH
conda create -y -n isaacteleop -c conda-forge --override-channels python=3.11
conda activate isaacteleop
pip install "isaacteleop[cloudxr,retargeters]==1.0.193" --extra-index-url https://pypi.nvidia.com
conda deactivate
```

에셋(손 6종 + core)은 NAS 레포에 이미 받아져 있다
(`scripts/asset_tools/download_robot_agents.py --all`, `scripts/asset_tools/download_assets.py --core`).

## 실행

### 터미널 1 — CloudXR 런타임 (계속 띄워 둠)

```bash
docker exec -it dexverse bash
/workspace/dexverse/DexVerse/scripts/teleop_tools/start_cloudxr_runtime.sh
# 최초 1회: NVIDIA CloudXR EULA 수락 프롬프트 [y/N] → y  (이후 /root/.cloudxr/run/eula_accepted 로 기억)
# "CloudXR runtime: running / WSS proxy: running" 이 뜨면 OK
```

스크립트가 처리하는 함정:
- `NV_CXR_ENABLE_PUSH_DEVICES=0` (없으면 런타임이 Push Hand Tracker를 잡아 손 관절이 계속 0/26)
- `PYTHONPATH` 해제 (Isaac Sim `setup_conda_env.sh`의 websockets 12가 끼면 "websockets >= 14" 로 죽음)
- 48322 포트 선점 검사 (다른 컨테이너의 런타임)

### 터미널 2 — DexVerse 텔레옵

```bash
docker exec -it dexverse bash
cd /workspace/dexverse/DexVerse
# 디버그 텔레옵 (저장 안 함)
scripts/teleop_tools/run_teleop.sh teleop_agent --task Dexverse-PickCube-v0 --robot_type floating_allegro_right
# 데모 녹화
scripts/teleop_tools/run_teleop.sh record_demos --task Dexverse-PickUpStick-v0 --dataset_dir grasping --num_demos 50
```

`run_teleop.sh`가 자동으로 붙이는 것 (직접 주면 그 값을 씀):
- `--teleop_device handtracking --enable_pinocchio --headless`
  - headless XR kit(`isaaclab.python.xr.openxr.headless.kit`)은 AR 세션을 자동 시작한다.
    `--gui`를 주면 headless를 빼고 VNC로 장면을 볼 수 있으며, 이때도 AR은 코드에서 자동 시작된다
    (viewport의 Start AR 버튼 불필요).
- teleop_agent에는 `--xr_stream_log 120`: 120프레임마다 손 스트림 요약
  `[xr] f=... R: wrist=(x,y,z) nonzero=26/26` (STALE = 손이 추적 범위 밖, 마지막 자세 유지 중)
- `/root/.cloudxr/run/cloudxr.env` 를 source 하고, conda python이 잡히지 않도록 kit python
  (`/workspace/isaaclab/_isaac_sim/python.sh`)을 직접 실행한다. 이 이미지의 대화형 셸은 conda base가
  켜져 있어서 `isaaclab.sh -p`가 conda python을 잡는다.

그 외 옵션은 upstream 그대로: `--teleop_retargeter relative|absolute`, `--retargeting_scheme dexpilot|vector`,
`--enable_debug_vis`, `--show_ranges`, `--seed`.

### Quest 3

1. Tailscale 켜고 `tailscale status`에 work1(100.108.68.0)과 quest-3가 같이 보이는지 확인.
2. 인증서 수락: Quest 브라우저에서 `https://100.108.68.0:48322` → 고급 → 계속.
3. `https://nvidia.github.io/IsaacTeleop/client` → Server IP `100.108.68.0`, Port `48322` → Connect → Enter VR.
4. 손추적: `chrome://flags`의 WebXR 실험 기능 ON(브라우저 재시작), 설정에서 손 추적 ON·컨트롤러 OFF,
   사이트 손추적 권한 허용.
5. 헤드셋 메뉴 **START** → 이때 손목 자세가 캘리브레이션되고 로봇이 따라 움직인다. STOP/RESET도 메뉴에서.

## 트러블슈팅

| 증상 | 확인 |
|---|---|
| `[xr] ... nonzero=0/26` 계속 | 런타임 로그(`/root/.cloudxr/logs/`)의 `Selected devices`가 Push Hand Tracker면 push=0 미적용 → 런타임 재시작. Quest WebXR flag / 손추적 / 권한 재확인 |
| `STALE` | 손이 추적 범위 밖. OpenXRDevice가 마지막 유효 자세를 유지하므로 로봇은 멈춰 있음 |
| Quest 클라이언트 "WebSocket timeout" / disconnected | 새 컨테이너는 WSS 자체서명 인증서를 새로 만든다 → Quest에서 `https://100.108.68.0:48322`를 다시 열어 "Certificate Accepted" 페이지까지 수락. 클라이언트 페이지의 로컬 네트워크 접근 권한 팝업이 뜨면 허용. 판별: `/root/.cloudxr/logs/wss.*.log`에 `Proxying ('100.103.54.31', ...)` 줄이 없으면 Quest 요청이 서버까지 안 온 것(Quest/네트워크 쪽 문제) |
| 브라우저 접속 중 `isaacsim.asset.browser` 확장 import 오류(idna) | 에셋 브라우저 UI 확장만 실패. 텔레옵과 무관하므로 무시 |
| `CloudXR runtime is not running` | 터미널 1의 런타임을 먼저 띄울 것 (`cloudxr.env`는 EULA 확인 전에 생성되므로 파일 유무가 아니라 48322 포트로 판단) |
| `port 48322 is already in use` | 다른 컨테이너(vls 등)의 CloudXR 런타임을 내릴 것 |
| `libcusparseLt.so.0: cannot open shared object file` (XR에서만) | 이미지가 옛 2.3.2-conda 빌드(torch 번들 손상). 수정된 `isaaclab-vnc:2.3.2-conda`로 컨테이너를 다시 만들 것 |
| START 전 손을 안 보여줬더니 로봇이 튐 | START는 그 순간의 손목 자세로 캘리브레이션한다. 손이 추적되는 상태에서 START |

## 변경 파일 (브랜치 `teleop/quest-cloudxr6`, feat/multi-hand-support 기반)

- `source/dexverse/dexverse/teleop_utils/xr_session.py` — AR 세션 자동 시작, 손 스트림 로거
- `scripts/teleop_agent.py` — 위 두 기능 연결, `--xr_stream_log N`
- `scripts/record_demos.py` — AR 세션 자동 시작
- `scripts/teleop_tools/start_cloudxr_runtime.sh`, `scripts/teleop_tools/run_teleop.sh`
- `docs/teleop_quest_cloudxr6.md` (이 문서)
