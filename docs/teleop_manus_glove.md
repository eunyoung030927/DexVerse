# DexVerse 텔레옵 — Manus 글러브 + Quest 3 컨트롤러

Manus 글러브 손가락 + 손등에 단 Quest 3 컨트롤러 손목으로 하는 VR 텔레옵·녹화는 저장소 루트의
`cross_dex` 저장소의 `cross_dex/teleop/` 패키지로 옮겼다 (2026-11-10, 이전: 이 레포의 `vr_teleop/`) (기기 처리·캘리브레이션·리타게팅을 시뮬레이터와 분리해 이식 가능하게).

- 전체 설명·사용 순서·환경변수: `codes/cross_dex/cross_dex/teleop/README.md`
- 캘리브레이션: `codes/cross_dex/cross_dex/teleop/tools/calibration/README.md`
- DexVerse 연결: `--teleop_device manus` (`scripts/teleop_agent.py`, `scripts/record_demos.py` → `cross_dex.dexverse.adapters.dexverse_integration`)
