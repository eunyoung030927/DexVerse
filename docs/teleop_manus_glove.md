# DexVerse 텔레옵 — Manus 글러브 + Quest 3 컨트롤러

Manus 글러브 손가락 + 손등에 단 Quest 3 컨트롤러 손목으로 하는 VR 텔레옵·녹화는 저장소 루트의
[`vr_teleop/`](../vr_teleop/README.md) 패키지로 옮겼다 (기기 처리·캘리브레이션·리타게팅을 시뮬레이터와 분리해 이식 가능하게).

- 전체 설명·사용 순서·환경변수: [`vr_teleop/README.md`](../vr_teleop/README.md)
- 캘리브레이션: [`vr_teleop/vr_teleop/tools/calibration/README.md`](../vr_teleop/vr_teleop/tools/calibration/README.md)
- DexVerse 연결: `--teleop_device manus` (`scripts/teleop_agent.py`, `scripts/record_demos.py` → `vr_teleop.teleop.dexverse_integration`)
