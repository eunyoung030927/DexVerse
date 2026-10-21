# 손가락 구동값: 공식 자료 vs DexVerse 설정, tactile 이상값

floating 손(오른손)의 손가락 PD 구동(강성·감쇠·토크 한계)을 제조사·공식 자료 값과 비교하고, 공식 값으로 바꿨을 때
tactile 이상값(패드 힘 > 100 N)이 얼마나 줄어드는지 정리한다 (2026-10-20, vast-isaac).

- 값 파일: `source/dexverse/dexverse/data_collection/official_finger_drives.json` (손별 출처·DexVerse 기본값·override)
- 적용 도구: `scripts/experiments/drive_override.py` (`DEXVERSE_DRIVE_OVERRIDE=@<hand>`로 record_demos.py 실행, 기본 설정은 그대로)
- 측정: `DEXVERSE_TACTILE_PROBE=<file.npz>` (물리 substep별 패드 힘·최소 접촉 거리) → `scripts/experiments/tactile_probe_analyze.py`
- 실험 기록: `experiments/vast-isaac/tactile-outlier-teleop-vs-retarget/` (공유 폴더)

**기본 로봇 설정(`robot_agents/<hand>/floating.py`)은 바꾸지 않았다.** 공식 값은 재생 실험에서만 적용했다.

## 1. 구동값 (단위: 강성 Nm/rad, 감쇠 Nm·s/rad, 토크 한계 Nm)

USD에 degree 기준으로 적힌 값(Isaac Sim 각도 구동)은 ×57.3으로 rad 기준 환산.

| 손 | 출처 | 강성 | 감쇠 | 토크 한계 | DexVerse 설정 (강성 / 감쇠 / 한계) |
|---|---|---|---|---|---|
| Allegro V4 | 강성·감쇠: Wonik 공식 ROS v4 드라이버 (τ = (kp·e − kd·v)/1200, kp 4000, kd 150~300). 토크 한계: 공개 사양 상한 | 3.33 | 0.125~0.25 | **0.9** (전 관절) ※ | 50 / 0.1 / 30 |
| Inspire RH56DFX | 제조사 모델 미공개. NVIDIA Isaac Sim 튜토리얼 5·6 (손끝 10 N × 0.084 m × 2) | 4.01 (0.07/deg) | 0.0057 (0.0001/deg) | 1.68 | 구동 20 / 2.0 / 5, 연동(mimic) 0 / 0 / 10 |
| Shadow | IsaacLab `SHADOW_HAND_CFG` (NVIDIA 자산, OpenAI J(k) → sr J(k+1)) | 1.0 | 0.1 | 0.72~2.37 | 10 / 0.1 / 10 |
| LEAP | 공식 LEAP_Hand_Sim (pgain 3, dgain 0.1, effort 0.5; URDF 0.95) | 3.0 | 0.1 | 0.5 | 20 / 1.0 / 3 |
| Wuji | 제조사 wuji-hand-description MJCF (관절별 kp, kv, forcerange) | 0.21~0.69 | 0.009~0.031 | 0.15~0.65 | 20 / 0.5 / 10 |
| Sharpa Wave | 제조사 sharpa-urdf-usd-xml (MITmode USD + URDF effort) | 0.62~23.6 | 0.012~0.39 | 0.19~3.3 | **제조사 값 그대로** (22개 관절 일치) |

- ※ Allegro 토크 한계 (2026-10-21 변경): 처음에는 Wonik 드라이버 PWM 상한(관절별 0.24~0.56 Nm)을 썼으나, 그 값으로는 0.5 kg 냄비(GraspPan)를 들지 못해(들고 있기 시험 3~4/4 낙하, 공식 구동 리타게팅 0/5) 공개 사양의 높은 값(0.7 Nm, 과구동 약 0.9 Nm) 중 0.9 Nm를 쓴다.
- Allegro·Inspire 공식 URDF의 effort(15, 10, 1)는 전 관절 같은 자리 표시용 값이라 쓰지 않았다.
- Wuji 설정 코드 주석은 "공식 MJCF 값"이라 하지만 현재 공식 저장소 값과 다르다.
- 강성·한계 배수 (DexVerse / 공식): Allegro 15배·33배, Inspire 5배·3배, LEAP 7배·6배, Wuji 29~95배·15~67배, Shadow 10배·4~14배.

### Sharpa 관절별 (공식 = DexVerse)

| 관절 | 강성 | 감쇠 | 토크 한계 | 공식 MJCF kp (MuJoCo용, 참고) |
|---|---|---|---|---|
| index MCP_FE | 4.02 | 0.157 | 1.864 | 4.76 |
| index MCP_AA | 5.37 | 0.160 | 1.864 | 6.62 |
| index PIP | 1.62 | 0.045 | 0.638 | 0.9 |
| index DIP | 0.62 | 0.012 | 0.189 | 0.9 |
| pinky CMC | 1.75 | 0.025 | 0.529 | 1.38 |
| thumb CMC_FE / CMC_AA | 13.56 / 13.38 | 0.380 / 0.355 | 3.3 | 6.95 / 13.2 |
| thumb MCP_FE / MCP_AA | 23.49 / 23.64 | 0.389 / 0.380 | 1.864 | 4.76 / 6.62 |
| thumb IP | 1.62 | 0.045 | 0.638 | 0.9 |

middle·ring·pinky의 MCP·PIP·DIP는 index와 같다.

## 2. 공식 구동값 적용 결과 (>100 N substep 비율, 최대 힘)

같은 데모를 기록 상태 재생(`--replay_states`)으로 돌리고 손가락 구동만 바꿨다.
Allegro 행은 변경 전 토크 한계(0.24~0.56 Nm)로 잰 값이다 (2026-10-21 이후 공식 값은 0.9 Nm).

| 손 (데이터) | DexVerse 설정 | 공식 구동값 |
|---|---|---|
| Allegro (텔레옵 HammerStrike, 10 ep) | 1.46% (258 N) | 0.00% (84 N) |
| Inspire (텔레옵 HammerStrike, 10 ep) | 5.87% (1278 N) | 1.07% (1210 N) |
| Shadow (원본 텔레옵 PourMug, 10 ep) | 0.75% (178 N) | 0.00% (98 N) |
| Wuji (리타게팅 PourMug, 10 ep) | 0.03% (166 N) | 0.00% (106 N) |
| LEAP (리타게팅 PourMug, 3 ep) | 5.45% (168,715 N) | 2.96% (130,539 N) |
| Sharpa (텔레옵 HammerStrike, 이미 공식 값) | 0.07% (233 N) | — |

- 손가락 구동 세기가 이상값의 주원인이다. 같은 데모에서 Allegro 한계 30→1~3 Nm 또는 강성 50→5로 1.46%→0.01~0.03%,
  Sharpa에 Allegro 구동을 넣으면 0.07%→15.4%. Inspire 감쇠 2.0→0.2는 영향 없음(6.2%).
- 남는 것(구동과 무관, 미확정): Inspire 약 1%는 상태 복원 직후 첫 substep에 몰려 재생 인공물로 추정.
  LEAP 10만 N대는 물리적으로 불가능한 값이라 리타게팅 상태 자체의 깊은 겹침으로 추정.

## 3. 주의

- 기록 상태 재생은 매 step 상태를 덮어쓰므로, 공식 구동값으로 태스크를 실제로 수행할 수 있는지는 확인하지 않았다
  (action 재생·평가 필요).
- 상태 복원(`scene.reset_to`)은 관절 속도 목표값을 그 순간 속도로 설정한다. 조건 간 비교는 유효하나 절대값은 영향받을 수 있다.
