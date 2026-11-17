# 기능적 손 자세(Functional Hand Posture) 스펙 v0 — cross-hand finger latent codec 페어 데이터용

작성: 261116 / 대상: sharpa, shadow, wuji (5지) + allegro, leap (4지 = 엄지 + 검지·중지·약지, 소지 없음) / 공유 latent z = 32-D

> 표기 규칙: **[V]** = 웹에서 직접 확인한 내용, **[M]** = 기억 기반(원문 미확인, 구현 전 검증 필요). 원문 PDF를 열람하지 못한 항목(Cutkosky, Kamakura 14종 이름, Elliott & Connolly 세부 명칭, Kapandji 랜드마크 번호)은 [M] 표시. 긴 인용 없이 요약만 함.

---

## 0. 한눈 요약 (결론)

1. 문헌의 공통 축은 5개: **(a) opposition type(pad/palm/side), (b) virtual finger(VF) 할당, (c) thumb abducted/adducted, (d) power–intermediate–precision, (e) 물체 크기·형상(aperture)**. 우리 스펙의 각 자세는 이 5축 + 손가락별 굴곡/외전 수준 + 접촉점으로 정의한다 → IK 타깃으로 변환 가능.
2. 정적 자세 **58개**(S 계열) + 동적 primitive **14개**(D 계열)로 구성. Feix 33종을 17개 hand-configuration 군으로 묶고(원논문도 17군 가능하다고 보고 [V]), 거기에 Kapandji 10-target(엄지 접촉 지도), Kamakura/Liu의 비-prehensile·일상 자세(point, press, hook, flat hand 등), 순수 kinematic 앵커(open, fist, spread)를 추가.
3. 4지(엄지+3) 손: 소지 역할은 **"ulnar-most finger 추상화"**로 처리 — 약지가 ulnar 지지/VF2 말단을 승계. 소지 고유 접촉이 필요한 자세(Kapandji 6–10, 소지 pinch, ring/fixed hook 4지판 등)는 **infeasible 마스크**로 두고 페어 손실에서 제외(드롭하지 말고 마스킹).
4. 커버리지: 각 자세를 이산 프로토타입이 아니라 (aperture, spread, flexion bias, contact offset) 4~5축 연속 변수로 Sobol 샘플 → IK → 손 크기 s로 정규화한 **손끝 위치 voxel 커버리지 + 엄지-손가락 pairwise 거리 공간 커버리지 + Kapandji 도달률 + (자세×크기) 셀 점유율** 4종 지표.

---

## 1. 문헌 요약표

| # | 출처 | 무엇을 분류 | 클래스 수 | 핵심 차원 | 링크 |
|---|---|---|---|---|---|
| 1 | Cutkosky 1989, IEEE T-RA 5(3):269–279 | 제조 현장 machinist의 파지 | 16 [M] (Feix 논문이 Cutkosky 16/17종을 33에 포함한다고 보고 [V]) | power ↔ precision 계층, 물체 크기·형상, 힘/dexterity 요구 | https://doi.org/10.1109/70.34763 · https://biorobotics.harvard.edu/pubs/1990/human%20grasp%20choice.pdf |
| 2 | Kamakura et al. 1980, Am J Occup Ther 34(7):437–445 | 정지 prehension(7명, 98개 물체) | 14 패턴 / 4 카테고리 [V] | 손가락 위치 + 접촉 면적, 4 범주(power, intermediate, precision, 비-prehensile 계열 [M]) | https://doi.org/10.5014/ajot.34.7.437 · https://www.jstage.jst.go.jp/article/jjrm1964/15/2/15_2_65/_article |
| 3 | Iberall/Bingham/Arbib 1986; Iberall & MacKenzie 1990 — opposition space, virtual finger | 파지의 기능 추상화 | opposition 3종(pad/palm/side) + VF 개념 [V] | VF = 한 기능으로 같이 움직이는 손가락 묶음(palm 포함), 힘 인가 방향 | https://www.cs.cmu.edu/afs/cs/user/sbk/www/thesis/opposition-space.html · https://pmc.ncbi.nlm.nih.gov/articles/PMC4656720 |
| 4 | Feix et al. 2016, IEEE THMS 46(1):66–77 | 정적·안정·단일 손 파지 | **33종**, 17 hand-config 군 [V] | opposition type, VF 할당, power/precision/intermediate, 엄지 abducted/adducted | https://ieeexplore.ieee.org/document/7243327 · https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf |
| 5 | Kapandji 1986, Ann Chir Main 5(1):67–73 (PMID 3963909) | 엄지 opposition 임상 점수 | 10 점수(타깃) [V] | 엄지 tip이 닿는 가장 먼 위치: 1=검지 근위지골 요측 … 3–6 손끝 pulp, 7–10 소지 쪽/손바닥 쪽(palm crossing) [V 요약, 랜드마크 세부는 이차자료]. 점수 6 이후 각도 증가가 포화한다는 보고 있음 | https://en.wikipedia.org/wiki/Kapandji_score · https://doi.org/10.1177/1753193419843837 |
| 6 | Elliott & Connolly 1984, Dev Med Child Neurol 26:283–296 | 손 내부(intrinsic) 조작 동작 | 3 클래스(simple synergy / reciprocal synergy / sequential pattern) + 하위 동작 ~ 10 [M: pinch, dynamic tripod, squeeze, twiddle, rock, radial roll, index roll, linear step, rotary step, interdigital step] | 손가락이 한 단위로 움직이는가, 교대로 움직이는가, 순차인가 | (원문 미열람) 이차: https://www.eng.yale.edu/grablab/pubs/Bullock_ICORR2011.pdf |
| 7 | Exner 1989 (OT, in-hand manipulation) | 아동 IHM 기술 | 3 유형(translation, shift, rotation), 이후 6 구성요소로 세분(finger-to-palm / palm-to-finger translation, simple·complex shift, simple·complex rotation) [V] | 물체 이동 방향: 손가락↔손바닥, 손끝 선형, 축 회전 | https://pmc.ncbi.nlm.nih.gov/articles/PMC4424153 · https://scielo.org.za/pdf/sajot/v46n2/09.pdf |
| 8 | Bullock, Ma, Dollar 2013, IEEE Trans Haptics 6(2):129–144 | 손-중심 manipulation(= in-hand) 분류 | prehension/접촉/물체 운동의 조합 분류(연속 코드) [V 요약] | 손 접촉 유지 여부, prehension 유형, 물체의 손 프레임 상대 운동(translation/rotation 등) | https://www.eng.yale.edu/grablab/pubs/Bullock_TOH2013_1.pdf · https://dblp.org/rec/journals/toh/BullockMD13 |
| 9 | Liu, Feng, Nakamura, Pollard 2014, Humanoids (+2015 chapter) | 일상 행동 중 파지 | 기존 분류에 없는 추가 자세 다수, 추가 속성 제안 [V] | 의도된 motion, force, stiffness를 분류 속성으로 추가 | https://www.homepages.ucl.ac.uk/~ucacyna/papers/TaxonomyEverydayGrasps.pdf · https://ri.cmu.edu/?p=119868 |
| 10 | Stival et al. 2019, J NeuroEng Rehab | 정량(EMG+kinematics) taxonomy | 20 grasp → **5 움직임 범주** [V] (40명) | 전체 모양·손가락 위치·근활성으로 계층 클러스터링 → 질적 분류를 부분 확인 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6377750/ |
| 11 | Santello, Flanders, Soechting 1998, J Neurosci 18(23):10105 | 포스처럴 시너지(PCA) | PC 2–3개가 큰 분산 설명 (논문은 "소수", 후속 Weiss&Flanders 2004는 PC2개≈80% [V]) | 시너지는 표준 grip taxonomy와 일치하지 않음 → 자세 모양은 힘 접촉과 분리 제어 | https://pmc.ncbi.nlm.nih.gov/articles/PMC6793309 · https://www.ece.uvic.ca/~bctill/papers/mocap/Weiss_Flanders_2004.pdf |
| 12 | Dexonomy (2025, arXiv 2504.18829) | Feix 33 중 31종을 관절형 로봇손(Shadow 등)에서 합성 | 31 grasp type, 9.5M grasp, 10.7k 물체 [V] | 손·type당 **사람 주석 템플릿 1개**에서 시작해 대규모 합성; type-conditioned 생성 | https://arxiv.org/abs/2504.18829 |
| 13 | RBO Hand 2 (soft hand) | Feix 33 중 31 실행 [V 요약] | 31/33 | 실물 로봇손에서 taxonomy 실행 사례 | (SAGE: https://doi.org/10.1177/0278364915592961) |
| 14 | DexGYS / DexGYSNet (NeurIPS 2024, arXiv 2405.19291) | 언어 안내 파지 | 5만 grasp–언어 쌍, 1.8k 물체 [V]; **taxonomy 라벨은 확인 못함** | 언어 의도 | https://arxiv.org/abs/2405.19291 |
| 15 | 크로스 임바디먼트 파지 계열: DRO-Grasp, GenDexGrasp(hand-agnostic contact map), CEDex(4종 gripper 20M), CrossDex(유니버설 사람손 공간→retarget), GraspGraphNet(URDF 그래프) | 손 간 파지 전이 | taxonomy 라벨 사용은 확인 못함 [V 부재] | contact map·robot–object 거리·URDF 그래프가 손-불변 표현 | https://arxiv.org/pdf/2410.02479 · https://arxiv.org/abs/2603.16806 · https://arxiv.org/pdf/2607.11031 |
| 16 | 로봇 엄지 Kapandji 평가 사례: RBO Hand 3, DexLink Hand, SyLink Hand, Anthro-Thumb(9/10) | 로봇 엄지 opposition 벤치 | 10 타깃 | 미리 기록한 자세 재생으로 타깃 접촉 점수화 [V] | https://arxiv.org/pdf/2201.10883 · https://arxiv.org/pdf/2606.17418 · https://arxiv.org/pdf/2606.14250 · https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1496073/epub |

관찰:
- **taxonomy 라벨을 cross-embodiment 학습에 직접 쓴 선행**은 검색에서 확인 못함(위 15행의 방법들은 contact/geometry 기반). Dexonomy가 "type 라벨 + 손별 템플릿"으로 가장 가깝다. → 우리 페어 세트 = "type 라벨된 hand-agnostic 기능 정의 → 손별 IK" 는 공백일 가능성 (선행조사 추가 필요).
- Santello는 "시너지 ≠ grip type"이라 보고: **synergy 공간만 샘플하면 grip-type 다양성이 보장되지 않는다** → 기존 play data의 한계를 문헌이 뒷받침.
- Kapandji 점수 6 이후 포화 보고: 7–10은 엄지 각도가 아니라 **손바닥 횡단(cross-palm) 도달성**을 측정 → 점수가 아닌 타깃 위치 자체를 커버리지 단위로 쓰는 게 낫다.

---

## 2. 기능적 정의 스키마

### 2.1 좌표·정규화
- 손 크기 **s** = (손목 원점 → 중지 끝, 완전 신전) 거리. 보조 W = 손바닥 폭(index MCP ~ ring/little MCP). 모든 aperture·거리는 s 단위. 5손 s를 URDF에서 계산해 저장 (손별 값은 측정 필요).
- 손가락 슬롯: T(엄지), I, M, R, L. 4지 손은 L 슬롯 없음.
- 단계 코드: 굴곡 **E**(0–10%) / **S**(10–35%) / **M**(35–65%) / **F**(65–100%), 관절 범위 정규화(손별 limit 기준). 외전 **N**(모음) / **m**(중간) / **W**(최대 벌림). 엄지 위치 **ABD**(외전: CMC 열림, 엄지가 손바닥 평면에서 떨어짐) / **ADD**(내전: 검지 측면에 붙음).
- Opposition: **pad**(손끝 pulp 대 pulp), **palm**(손바닥 대 손가락 패치), **side**(손가락 측면 — 엄지 pulp 대 검지 요측 등).
- 접촉 약어: tip(말단 pulp), DP/MP/PP(말단/중간/근위 지골), Pm(palm), Th(엄지), VFk = virtual finger k. VF1 = 보통 엄지(또는 palm), VF2 = 반대편을 형성하는 손가락 묶음, VF3 = 보조(있을 때).
- IK 타깃 변환: 각 자세 → (a) 접촉 쌍 목록(손가락 링크 pad 프레임 간 거리 ≤ ε, 법선 마주보기), (b) 비접촉 손가락의 굴곡/외전 수준(관절 범위 비율 사전 prior), (c) 가상 물체(구/원기둥/박스/디스크, 크기 d/s) 표면 접촉 제약. 접촉 위치는 phalanx pad 프레임을 URDF에서 손별로 정의.

### 2.2 ID 규칙
S-P(power), S-I(intermediate), S-R(precision), S-K(Kapandji), S-N(non-prehensile/기하 앵커), S-X(일상·추가). 괄호 = Feix 번호 대신 Feix 명칭.

---

## 3. 정적 자세 목록 (표에 68개 나열 = 코어 58 + 확장 10, 아래 합계 참조)

열: 정의 요약 | 참여 손가락·VF | opp | 엄지 | 접촉 | aperture(÷s) | 굴곡/외전 | 4지 처리·infeasible.
**4지 처리 열 약어**: "R→ulnar" = 약지가 소지 역할(ulnar 지지)을 승계, "–L" = 소지 제외로 정의 가능, "✗4" = 4지 손 infeasible(마스크), "✗sh" = 특정 손 제약(§6).

### 3.1 Power (Feix: palm opposition은 VF1=palm, pad opposition은 VF1=엄지; VF 숫자 2–5 = 검지–소지) — 16 (Feix 대조 수정: 기존 제목 'VF1=엄지' 오류)

| ID | 이름(Feix 대응) | 참여/VF | opp | 엄지 | 접촉 | aperture | 굴곡/외전 | 4지 처리 |
|---|---|---|---|---|---|---|---|---|
| S-P01 | Large diameter cylinder (Large Diameter) (Feix 대조 수정) | T+IMRL / VF1=palm, VF2-5 (Feix; 엄지는 wrap 참여) | palm | ABD | 4지 PP/MP/DP 패드, Pm, 엄지 pad | 0.35–0.55 | I–L: S→M, 외전 W | R→ulnar, 폭이 L 없이 작아져 상한 aperture 감소 |
| S-P02 | Small diameter cylinder (Small Diameter) (Feix 대조 수정) | T+IMRL / VF1=palm, VF2-5 (Feix; 엄지는 wrap 참여) | palm | ABD | 4지 DP/MP, Pm, 엄지 pad가 검지 위 | 0.12–0.25 | M→F, 외전 N | R→ulnar |
| S-P03 | Medium wrap (Medium Wrap) (Feix 대조 수정) | T+IMRL / VF1=palm, VF2-5 (Feix; 엄지는 wrap 참여) | palm | ABD | 4지 + 엄지 wrap, 엄지 tip이 중지 근처 | 0.20–0.35 | M, N–m | R→ulnar |
| S-P04 | Adducted-thumb wrap (Adducted Thumb) (Feix 대조 수정) | IMRL (Feix VF2-5; VF1=palm, 엄지는 측면 지지) | palm | ADD | 4지 wrap, 엄지는 검지 측면에 얹음 | 0.15–0.30 | M→F, N | R→ulnar |
| S-P05 | Light tool / hammer-like (Light Tool) (Feix 대조 OK) | T+IMRL | palm | ADD | 4지 wrap, 엄지 길이방향 | 0.15–0.25 | M→F, N | R→ulnar |
| S-P06 | Power sphere (Power Sphere) (Feix 대조 OK) | T+IMRL | palm | ABD | 5지 전체 + Pm | 0.45–0.80 | S→M, 외전 W | R→ulnar; 큰 구 상한 축소 |
| S-P07 | Power disk (Power Disk) (Feix 대조 수정) | T+IMRL / VF1=palm, VF2-5 (Feix; 엄지는 wrap 참여) | palm | ABD | 5지 + Pm, 평평한 디스크 | 0.50–0.80 | S, W | R→ulnar |
| S-P08 | Sphere 4-finger (Sphere 4-Finger) (Feix 대조 수정) | T+IMR / VF1=T, VF2-4 (Feix #26; L 불필요) | pad (Feix: palm 아님) | ABD | 4 손가락 패드 + Pm | 0.35–0.60 | M, m | 소지 불필요 → 4지 손 OK (R→ulnar 불필요) |
| S-P09 | Palmar grasp / ball-in-palm (Palmar) (Feix 대조 수정) | T+IMRL | palm | ADD (Feix #30; 기존 ABD 오류) | Pm 지배, 손가락 약하게 닫음 | 0.20–0.40 | S–M | OK |
| S-P10 | Ring grasp (Ring) (Feix 대조 수정) | T+I / VF1=T, VF2=I (Feix #31 Ring) | pad (Feix: palm 아님) | ABD | 엄지-검지가 고리, Pm 약함 | 0.10–0.30 | I:M, T:M, 나머지 F | –L, OK |
| S-P11 | Fixed hook (Fixed Hook) (Feix 대조 수정) | IMRL (T 없음) | palm (Feix #15) | ADD | 4지 중-말단 지골이 갈고리, 엄지 이완 | 0.05–0.20 | PIP F, MCP E–S | R→ulnar |
| S-P12 | Stick / baton (Stick, Feix #29) (Feix 대조 수정) | T+I / VF1=T, VF2=I (Feix: VF2) | side | ADD | 엄지 pad - 검지 측면으로 막대(펜형) 지지, 분류=Intermediate(ID 접두 P는 유지) | 0.10–0.20 | M, N | OK (4지 손 가능) |
| S-P13 | Lateral (Feix #16, S-I01과 중복) (Feix 대조 수정) | T+I side | side | ADD | 엄지 pad - 검지 요측; 분류=Intermediate(Feix), VF2 | 0.02–0.10 | I:F-M, 나머지 F | OK |
| S-P14 | Parallel-extension on flat power (Extension Type) (Feix 대조 수정) | T+IMR / VF1=T, VF2-4 (Feix #18 Extension Type) | pad | ABD (기존 ADD 오류) | 손가락 신전, 엄지와 손가락 pad 사이로 얇은 판 | 0.03–0.12 | E, N | 소지 불필요 → OK |
| S-P15 | Index-finger extension (Index Finger Extension) (Feix 대조 수정) | IMRL 중 MRL wrap / VF3-5 (Feix #17), 검지 VF2 비접촉 | palm | ADD (기존 ABD/ADD 모호) | 3지 wrap, 검지 연장(도구 방향 가리킴) | 0.12–0.25 | I:E, MRL:M–F | MR만 wrap (L 없음) |
| S-P16 | Distal power wrap (Distal Type, Feix #19) (Feix 대조 수정) | T+IMRL / VF1=T, VF2-5 | pad (Feix: palm→pad 아님) | ABD | 4지 말단 위주 | 0.10–0.20 | M→F, N | OK |

### 3.2 Intermediate (side opposition) — 6 (Feix 대조: Feix Intermediate는 #16, 23, 25, 29, 32, 21 — 아래 I05는 Feix상 Precision, P12·P13은 Feix상 Intermediate)

| ID | 이름 | 참여/VF | opp | 엄지 | 접촉 | aperture | 굴곡/외전 | 4지 처리 |
|---|---|---|---|---|---|---|---|---|
| S-I01 | Lateral pinch (key) (Lateral pinch) (Feix 대조 OK) | T+I | side | ADD | 엄지 pad - 검지 측면(MP) | 0.02–0.08 | I:S–M, 나머지 F | OK |
| S-I02 | Lateral tripod (Lateral Tripod) (Feix 대조 수정) | T+M / VF1=T, VF3 (Feix #25: VF3만) | side | ADD | 엄지 pad - 중지 측면(검지는 Feix상 비접촉; 기존 '검지 측면+중지 측면'은 Feix 외 변형) | 0.03–0.10 | M | OK |
| S-I03 | Stick between fingers (Feix 외 변형; Feix #29 Stick과 다름 — #29=엄지-검지 side) (Feix 대조 수정) | I,M (VF2=I+M) + T | side | ADD | 검지·중지 측면에 물체 끼움; Feix 직접 대응 없음(Adduction Grip #23과 인접) | 0.02–0.06 | M–F | OK |
| S-I04 | Adduction grip (Adduction Grip) (Feix 대조 수정) | I,M 측면 / VF2 (Feix #23) | side | ABD (Feix #23; 기존 'ADD/자유' 오류, 엄지는 비접촉) | 검지-중지 측면 | 0.01–0.04 | E–S | OK |
| S-I05 | Writing tripod (Writing Tripod) (Feix 대조 수정) | T+M / VF1=T, VF3 (Feix #20) | side (Feix #20; pad/side 혼합 아님) | ABD | 엄지 pad - 중지 측면(+검지 지지); 분류=Precision(Feix), ID 접두 I는 유지 | 0.02–0.06 | I:M, M:M, RL F | –L, OK |
| S-I06 | Tripod variation (Tripod Variation) (Feix 대조 수정) | T+M+R / VF1=T, VF3-4 (Feix #21) | side | ABD (기존 ABD/ADD 모호) | 엄지 pad - 중지·약지 측면(VF3-4) | 0.03–0.08 | M | OK |

### 3.3 Precision (pad opposition, VF1=T, VF2=손가락 묶음) — 18

| ID | 이름 | 참여/VF | opp | 엄지 | 접촉 | aperture | 굴곡/외전 | 4지 처리 |
|---|---|---|---|---|---|---|---|---|
| S-R01 | Tip pinch (Tip Pinch) (Feix 대조 OK) | T+I | pad(tip) | ABD | 엄지 tip - 검지 tip | 0.005–0.05 | I:M, T:M, 나머지 F/중립 | OK |
| S-R02 | Palmar pinch (Palmar Pinch) (Feix 대조 OK) | T+I | pad(pulp) | ABD | 엄지 pulp - 검지 pulp(관절 더 굴곡) | 0.01–0.08 | I:M–F, 나머지 F | OK |
| S-R03 | Inferior pincer (Inferior Pincer) (Feix 대조 OK) | T+I | pad(큰 pulp) | ABD | 엄지-검지 pulp, 큰 접촉 | 0.03–0.10 | I:F, T:F | OK |
| S-R04 | Tripod (Tripod) (Feix 대조 OK) | T+I+M | pad | ABD | 엄지 tip - 검지·중지 tip | 0.02–0.10 | M | OK |
| S-R05 | Quadpod (Quadpod) (Feix 대조 OK) | T+I+M+R | pad | ABD | 엄지 - 3지 tip | 0.03–0.12 | M, m | OK (4지 최대 pad 구성) |
| S-R06 | Five-pad pinch (Feix 외 확장; 가장 가까운 Feix=#13 Precision Sphere VF2-5, #28 Sphere 3 Finger는 Power) (Feix 대조 수정) | T+IMRL | pad | ABD | 엄지 - 4지 tip 모음 | 0.03–0.12 | M | R→ulnar; L 접촉은 ✗4 |
| S-R07 | Sphere 3-Finger (Feix #28; Feix 분류=Power·pad·VF2-3; 본 행의 precision 구는 확장 변형) (Feix 대조 수정) | T+I+M | pad | ABD | 3지 tip 구 | 0.20–0.45 | S–M, m | OK |
| S-R08 | Sphere 4-Finger (Feix #26; Feix 분류=Power·pad·VF2-4, S-P08과 중복; 본 행의 precision 구는 확장 변형) (Feix 대조 수정) | T+I+M+R | pad | ABD | 4지 tip 구 | 0.25–0.55 | S–M, m–W | OK |
| S-R09 | Precision disk (Precision Disk) (Feix 대조 OK) | T+IMRL | pad | ABD | 5지 tip 원반 | 0.25–0.55 | S, W | R→ulnar |
| S-R10 | Precision sphere 2-finger (Feix 외 변형, 대응 번호 없음) (Feix 대조 수정) | T+I | pad | ABD | 엄지-검지 tip 구 | 0.15–0.40 | S–M | OK |
| S-R11 | Prismatic 4-finger (Prismatic 4 Finger) (Feix 대조 OK) | T+IMRL | pad (VF2=IMRL) | ABD | 4지 pad 한 줄 + 엄지 | 0.02–0.10 | I–L:M, 엄지 대향 | R→ulnar |
| S-R12 | Prismatic 3-finger (Prismatic 3 Finger) (Feix 대조 OK) | T+IMR | pad | ABD | 3지 pad 줄 + 엄지 | 0.02–0.10 | M | OK |
| S-R13 | Prismatic 2-finger (Prismatic 2 Finger) (Feix 대조 OK) | T+IM | pad | ABD | 2지 pad + 엄지 | 0.02–0.08 | M | OK |
| S-R14 | Parallel extension (Parallel Extension) (Feix 대조 수정) | T+IMRL / VF2-5 (Feix #22) | pad(pulp) | ADD (Feix #22; 기존 ABD 오류) | 손가락 평행 신전, 엄지-4지 얇은 판 | 0.02–0.08 | E–S, N | R→ulnar |
| S-R15 | Pad-to-side pinch (Feix 외 변형) (Feix 대조 수정) | T+I | side/pad | ADD | 엄지 pulp - 검지 MP측면 | 0.01–0.05 | I:M | OK (Kamakura "side" 계열 [M]) |
| S-R16 | Distal pad grasp, 얇은 물체 (Feix 외 변형) (Feix 대조 수정) | T+I | pad(tip) | ABD | 엄지-검지 tip, 평평한 면 | 0.003–0.02 | I:S, T:S | OK |
| S-R17 | Small-ball pinch (Feix 외 변형) (Feix 대조 수정) | T+I(+M) | pad(tip) | ABD | tip - tip(+M) | 0.01–0.04 | M–F | OK |
| S-R18 | Extension pinch (Feix 외 변형) (Feix 대조 수정) | T+I | pad(tip) | ABD | 신전된 검지 tip - 엄지 tip | 0.05–0.20 | I:E–S | OK |

### 3.4 Kapandji 엄지 지도 (엄지 tip 타깃; 손가락은 중립/약간 굴곡) — 10
랜드마크 1–10: 이차자료(Wikipedia Kapandji score, 2026-11-16 확인)와 일치 확인 — K01–K10 전 행 (Kapandji 대조 OK, 이차자료 기준; 1986 원논문 미열람). 'K10 = 원위 손바닥 주름(소지 쪽)' 등 표기 일치.

| ID | Kapandji | 타깃(엄지 pad가 접촉) | opp | 엄지 | 굴곡/외전 | 4지 처리 |
|---|---|---|---|---|---|---|
| S-K01 | 1 | 검지 근위지골 요측 | side | ADD→약ABD | 4지 E–S | OK |
| S-K02 | 2 | 검지 중간지골 요측(PIP 부근) | side | ADD→ABD | 〃 | OK |
| S-K03 | 3 | 검지 tip | pad | ABD | I:S–M | OK |
| S-K04 | 4 | 중지 tip | pad | ABD | M:S–M | OK |
| S-K05 | 5 | 약지 tip | pad | ABD(+opposition 강) | R:S–M | OK |
| S-K06 | 6 | 소지 tip | pad | ABD+강한 opposition | L:S–M | **✗4**, shadow/sharpa/wuji에서는 가능 여부 별도 확인(✗sh: 소지 CMC·palm-arch) |
| S-K07 | 7 | 소지 DIP 주름 | pad/palm 경계 | 강한 cross-palm | L:M | ✗4 |
| S-K08 | 8 | 소지 PIP 주름 | palm | 〃 | L:M | ✗4 |
| S-K09 | 9 | 소지 MCP 주름(소지 근부) | palm | 〃 | L:M–F | ✗4 |
| S-K10 | 10 | 원위 손바닥 주름(소지 쪽 palm) | palm | 최대 cross-palm | 4지 E–M | **✗4 (약지 쪽 palm 점으로 "K10′" 대체 정의 가능)** |

→ 4지 손은 **K1–K5 + 대체 타깃 K9′(약지 MCP 주름), K10′(약지 아래 palm 점)**으로 평가. 타깃 점은 URDF 링크 프레임 상의 손-정규화 좌표로 정의(손가락 길이에 비례).

### 3.5 비-prehensile / 기하 앵커 / 일상 — 18

| ID | 이름(출처) | 참여 | opp | 엄지 | 설명 | aperture | 굴곡/외전 | 4지 처리 |
|---|---|---|---|---|---|---|---|---|
| S-N01 | Open flat hand | 전부 | – | ADD | 평평한 손 | – | E, N | OK |
| S-N02 | Open spread | 전부 | – | ABD | 최대 벌림 | tip 간격 최대 | E, W | L 없음(OK 4지 정의) |
| S-N03 | Fist | 전부 | palm | ADD | 주먹, 엄지 외측/검지 위 | – | F, N | OK |
| S-N04 | Fist, thumb extended(엄지 척) | IMRL | – | ABD | 주먹 + 엄지 신전 | – | F, T:E | OK |
| S-N05 | Pointing (index) | I | – | ADD | 검지 신전, 나머지 F | – | I:E, MRL:F | OK (Liu 일상 [V 속성]) |
| S-N06 | Two-finger point/"V" | I,M | – | ADD | 검지·중지 신전, 벌림 | – | IM:E, W | OK |
| S-N07 | Three-finger extension (3) | I,M,R | – | ADD | 3지 신전 | – | E, m | R→ulnar(그대로 OK) |
| S-N08 | OK sign | T+I pad, MRL 신전 | pad | ABD | 엄지-검지 원, 나머지 신전 | 0.05–0.15 | I:M, MRL:E | MRL → MR, L 제거 OK |
| S-N09 | Hook / claw(손가락 갈고리) | IMRL | – | ADD | 손가락만 굴곡, 손바닥 비접촉 | – | MCP E–S, PIP F | OK |
| S-N10 | Flat palm press (손바닥 누르기) | 전부 | palm | ADD | 평평한 손바닥 + 지면 | – | E, N | OK |
| S-N11 | Finger press / button press (검지 tip) | I | – | ADD | 검지 tip 하방 수직 | – | I:S, 기타 F | OK |
| S-N12 | Thumb press (엄지 tip 하방) | T | – | ABD | 엄지 단독 | – | – | OK |
| S-N13 | Scoop / cup (손 오목) | 전부 | palm | ABD | 손바닥 오목, 손가락 약굴곡 | – | S, N | R→ulnar |
| S-N14 | Pinch-through-gap (손가락 사이 끼움, 엄지 없음) | I,M | side | ADD | 검지-중지 사이 가는 물체 | 0.01–0.03 | S | OK (S-I04과 중복 → 제거 후보) |
| S-N15 | Palm-up present (손바닥 위 얹기) | 전부 | palm | ADD | 평평하게 물체 얹기, 약한 접촉 | – | E–S | OK |
| S-N16 | Grip-and-twist 준비 자세(미닫이 병뚜껑) | T+I+M | pad/side | ABD | 3지 pad 둥글게 | 0.15–0.30 | S–M | OK |
| S-N17 | Snap/flick 준비(엄지 위 중지 로딩) | T+M | pad | ABD→ADD | 엄지 tip - 중지 pad(압력), 나머지 F | 0.01–0.03 | M:F | OK |
| S-N18 | Counting 1–5 (손가락 개수 조합 5종) | 다양 | – | – | 손가락 e/f 서브셋 32조합 중 대표 | – | E/F 이진 | 4지는 해당 조합 마스크(L 의존) |

합계: 16 + 6 + 18 + 10 + 18 = **68**. (S-N14는 S-I04와 중복, S-P16/S-N08/S-R06 등 경계 자세는 1차 라운드에서 제외 가능 → 최소 코어 세트 58개 = 68에서 S-N14, S-N18, S-P16, S-R18, S-R17, S-P14, S-N15, S-R16, S-I06, S-R15 제외.) 본 스펙은 위 코어 58개 + 선택 10개를 "확장 세트"로 부름.

주 (261116 Feix 대조 완료): Feix 2016 Fig. 4(원논문 PDF)와 Feix 파생 행을 대조함. Feix 33종 중 31종이 본 표에 존재하며 **누락 2종: #13 Precision Sphere(pad, ABD, VF2-5, Precision), #32 Ventral(side, ADD, VF2, Intermediate)** — 추가 필요(S-R06을 #13에 재대응하거나 신규 행). 이름 옆 '(Feix 대조 OK/수정)' 표시가 대조 결과. 표의 VF 표기는 Feix 방식(숫자 2–5 = 검지–소지 / 엄지 VF1 또는 palm VF1)을 기준으로 함. Feix 분류(power/intermediate/precision)와 ID 접두(P/I/R)가 어긋난 행(P12·P13=Intermediate, I05=Precision, R07·R08=Power)은 ID 안정성을 위해 접두를 유지하고 행에 분류를 명시함. 'Feix 외 변형' 표기 행은 Feix 33종에 없는 확장.

---

## 4. 동적 in-hand primitive (14)

각 항목: 시작 상태 → 끝 상태 (기능 상태 = §3 자세 ID + 접촉 유지 조건). 모든 primitive는 "궤적 = 시작-끝 사이 N개 waypoint(기본 N=8) + 접촉 유지/해제 마스크"로 정의. 가상 물체(구/원기둥/막대, 크기 d/s)와 함께 정의하여 접촉 일관성 검증.

| ID | 이름(출처) | 시작 → 끝 | 참여 | 필요 DOF/제약 | 4지 | 
|---|---|---|---|---|---|
| D01 | Pinch open/close (Elliott: pinch) | S-R01(열린 aperture 0.15) ↔ S-R01(닫힘, 접촉) | T+I | 기본 | OK |
| D02 | Dynamic tripod (Elliott) | S-R04 열림 ↔ 닫힘, 3지 동기 | T+I+M | 기본 | OK |
| D03 | Squeeze (Elliott: palmar squeeze) | S-P09 ↔ S-P03(압착), palm 방향 | T+IMRL | 기본 | R→ulnar |
| D04 | Rock (Elliott: reciprocal) | 물체 일축 회전: 엄지 전진·검지 후퇴(side/pad 접촉 유지) | T+I | 엄지 2 DOF 이상 | OK |
| D05 | Twiddle (Elliott: reciprocal) | S-R04, 엄지+검지/중지 번갈아 pad rolling으로 물체 회전 | T+I(+M) | 엄지 외전축 | OK |
| D06 | Radial/ulnar roll (Elliott: radial roll, index roll) | 엄지-검지 pad 사이에서 물체 roll | T+I | 접촉 pad 유지 | OK |
| D07 | Linear step (Elliott: sequential) | 막대를 손가락 pad 사이에서 한 칸씩 이동(I→M→R) | T+IMR | 엄지 고정 지지, 손가락 교번 | R→ulnar (3지 step; 4지 1단계 단축) |
| D08 | Rotary step (Elliott: sequential) | 병뚜껑류 회전: T+I+M 번갈아 회전, regrasp | T+I+M | 엄지 외전/굴곡, 손가락 abduction | OK (제한적: leap/allegro abduction 한계) |
| D09 | Finger-to-palm translation (Exner) | S-R11 → S-P09(물체가 손바닥으로) | T+IMR | 연속 | OK |
| D10 | Palm-to-finger translation (Exner) | S-P09 → S-R04 | T+IMR | 연속 | OK |
| D11 | Simple shift (Exner) | S-R02 유지, 엄지-검지 pad가 물체 축 방향 선형 이동 | T+I | 엄지 슬라이딩(마찰 의존) | OK |
| D12 | Complex shift (Exner) | 2지 이상 손가락이 서로 다른 속도로 이동, 물체 밀기 | T+I+M | 위와 동일 | OK |
| D13 | Simple rotation (Exner) | 엄지-검지 사이 막대 90° 축 회전 (pencil spin) | T+I | 엄지 2 DOF + 검지 | OK |
| D14 | Complex rotation (Exner) | 180°/ flip, 접촉 해제 포함 (펜 뒤집기) | T+I+M | 접촉 해제/재접촉 | 상당 제한(✗sh: allegro 불가 가능성 높음) |

보너스(옵션 D15): **Kapandji sweep** — 엄지 tip이 K1→K10을 연속 이동(소지 쪽은 4지에서 K5′까지). **D16**: finger gaiting(손가락 교번 재파지, Bullock 분류 중 in-hand + 접촉 변화). 위 14개가 1차 목표.

Bullock 2013 분류에서의 활용: 각 primitive를 (접촉 유지 여부, 물체 운동 유형: translation/rotation/…)으로 라벨하면 §5 커버리지의 "motion-type 셀" 지표로 쓸 수 있음.

---

## 5. 4지(엄지+3) 손 매핑

1. **정의를 슬롯이 아닌 역할로 기술**: "ulnar-most finger" = 5지 손은 소지(L), 4지 손은 약지(R). VF2의 "말단 접촉" 요구는 역할로 매핑.
2. **Power wrap(P01–P09)**: 소지의 힘 기여·접촉이 없으므로 같은 물체 크기에서 **wrap 폭이 짧아짐** → aperture 상한을 손별 W 비율로 축소(5지 대비 약 ~0.75× [M, 근사]). 약지 위치를 L 쪽으로 이동하는 외전 보정(손별 허용 범위 내).
3. **Prismatic 4지(R11)**: 4지 손에서는 R12(3지)로 치환. **Quadpod/5-pad(R05, R06)**: 4지 손은 R05에서 최대.
4. **소지 의존 항목**: Kapandji 6–10, 소지 단독 접촉(S-N18 일부), hook의 소지 기여, "소지 측 palm 접촉" → ✗4 마스크. K9′/K10′는 약지 기준 대체.
5. **Virtual finger 이론으로 정당화**: VF는 "같이 움직여 하나의 기능을 수행하는 손가락 묶음"이므로 [V], 손가락 수가 바뀌어도 VF 기능(예: VF2=ulnar side 지지)이 유지되면 동일 기능 자세로 간주 가능. 단, **힘 용량(force capacity)은 달라진다**는 점은 보존되지 않음(위치 기능만 보존).
6. 손가락 수 부족 보정에 대한 문헌: Feix의 33종 중 하위 군은 **VF 구성만 같으면 손가락 수가 다른 변형(Prismatic 2/3/4, Sphere 3/4)**으로 명시적으로 나뉘어 있음 [V 개념]. → 우리 4지 대응은 Feix가 이미 정의한 "손가락 수 변형"을 그대로 사용 (5지: R11·R08, 4지: R12·R07).

### 5.1 손 클래스별 feasibility 요약

| 클래스 | 항목 | 비고 |
|---|---|---|
| 5지 (sharpa, shadow, wuji) 전부 가능 후보 | S-P01–P16, I, R, K1–K10, N | 소지 CMC 유무(shadow의 palm-arch 관절, sharpa/wuji는 [M])에 따라 K6–K10 도달성 편차 |
| 4지 (allegro, leap) 불가 | S-K06–K10, S-N18 일부, S-P11(소지 기여 필요 시), D14 | K9′·K10′ 대체; 엄지 opposition range가 작아 K5 이상은 별도 확인 필요 |
| 엄지 DOF 제한 손 | opposition 정도 요구 자세(S-K05, S-R06, D04/D05/D13) | §6.2 |
| 손가락 abduction 제한 손 | W 외전 자세(S-N02, S-P01 외전 W), 손가락 사이 side 파지(S-I03, S-I04) | allegro 손가락은 abduction이 매우 작음 [M] |

---

## 6. 커버리지 설계

### 6.1 연속 변이 샘플링 (프로토타입 주변)
각 자세 i마다 파라미터 벡터 θ = (a, σ, β, c, r):
- **a**: aperture/물체 크기 ∈ [a_min, a_max] (표의 범위), 로그 균등 or Sobol.
- **σ**: finger spread(손가락 간 외전) ∈ {N…W} 연속 [0,1].
- **β**: 굴곡 바이어스(손가락별 굴곡 수준에서 ±1 단계 이내 jitter), 손가락별 독립.
- **c**: 접촉 위치 오프셋 — pad 중앙 ↔ pad 모서리 / 지골 따라 근위↔말단 이동(예: tip→DP 중간→DP 근부, 각 ±0.3 pad 길이).
- **r**: 물체 형상(구/원기둥/박스/디스크)과 방향 롤(손목 프레임 기준 ±30°).
→ 자세당 Sobol 128–512 샘플(자세 난이도에 따라). 손별로 IK 풀이 후 **valid 필터**: (i) 접촉 오차 ≤ ε s (ε=0.01–0.02), (ii) 관절 한계 여유 ≥ 3%, (iii) 자기 충돌·관통 없음, (iv) 가상 물체 관통 < 1mm 상당(손별 스케일).
- 샘플은 **모든 5손에서 동일 θ**로 풀되, 푼 결과가 실패하면 **그 손만 마스크**(페어 손실에서 해당 손 항 제외). 최소 유효 손 수 ≥ 3이어야 페어로 채택 (엄지 의존 항목 때문).

### 6.2 자세 간 보간
- **Kapandji 경로**: S-K03→K04→K05(→K06) 등 인접 타깃 사이 엄지 tip을 손-정규화 공간에서 직선 보간 + IK(연속 warm-start), 이후 손가락 굴곡을 아울러 보간.
- **Grasp-type 보간**: 같은 VF 구성 내(예: R02↔R04↔R05 pad 수 증가, P03↔P02 aperture 감소) 직선 보간. **opposition type을 가로지르는 보간**(power↔precision, pad↔side)은 중간점이 비현실적일 수 있으므로 **접촉 연속성 체크**를 통과하는 샘플만 채택.
- **Synergy 앵커 연결**: open↔fist(Santello PC1 방향 [V: 시너지 개념]) 위에 위 자세들을 올려 "시너지 직선에서 벗어난 점들"을 의도적으로 추가(Santello가 시너지 ≠ grip taxonomy라고 한 것 [V]을 근거).
- 동적 primitive는 시작·끝 상태를 §3 자세 ID로 고정하고 중간은 접촉 제약을 유지하는 IK(waypoint 8–16).

### 6.3 커버리지 지표 (모두 계산 가능)

1. **C1 손끝 위치 voxel 커버리지**: 각 손가락(T,I,M,R[,L])의 손끝 위치를 손목 프레임에서 s로 정규화 후 0.05 s 격자로 voxel화. 지표 = 사람(또는 "최대 달성 reachable set"을 손 FK 몬테카를로로 구한 값) 대비 점유 비율. 손별 계산 후 **5손 평균 및 min** 보고. 동일 격자에서 sample density의 엔트로피도 보고(편중 감시).
2. **C2 엄지–손가락 pairwise 거리 공간**: 벡터 (|T–I|, |T–M|, |T–R|, |I–M|, |M–R| ÷s) (5지는 L 포함) 의 4–5차원 공간에서 격자 점유율 + 가장 가까운 샘플까지의 최대 거리(covering radius). 손 간 비교 가능(엄지 접촉 기하를 정규화).
3. **C3 Kapandji/opposition 도달률**: 타깃 k(1..10, 4지는 1..5+K9′,K10′)에 대해 엄지 tip이 타깃 중심 ≤ ε s 내 도달한 샘플 비율, 그리고 각 타깃 주변 **접근 방향(법선과 엄지 pad 법선 각도) 분포** 커버리지. 손별 Kapandji 점수 = 최대 도달 k.
4. **C4 자세×크기 셀 점유**: 셀 = (S-ID, a-bin(4–6분위)) — 각 셀 ≥ N_min(예: 8) 유효 샘플. 손별/교집합 보고. 교집합(5손 공통 유효 샘플 수)은 **페어 학습 가능량**의 직접 지표.
5. **C5 (선택) latent 측 지표**: 샘플을 인코딩한 z의 k-NN 커버리지(z 공간 점유 대비 사람 hand-keypoint 분포), 손 간 pair 거리(MPJPE in s 단위) — 코덱 평가용.

### 6.4 가중·샘플링 균형
- 클래스 균형: Feix 군 17개와 Kapandji 10과 동적 14를 **층화**(층당 동일 샘플 상한). power/precision/intermediate 비율은 대략 5:5:2 이상으로 정규화(쉬운 power 쏠림 방지).
- play data(랜덤/시너지)는 **기존 코덱 학습에 보조(<30%)**로 유지, taxonomy 세트를 앵커로.

---

## 7. 함정(Pitfalls)과 선행 처리

1. **운동학 차이**:
   - 같은 "tip pinch"라도 손별로 엄지 CMC 구성이 달라 **엄지 pad 접촉 가능 면적/법선이 상이** → 접촉을 "pad 프레임 간 거리+법선 반대" 제약으로 정의하고 관절각 직접 일치를 요구하지 않음.
   - 손 크기 s 정규화가 필수; 그러나 크기비가 다른 손(예: allegro 큼)에서는 작은 물체(a<0.02)가 손끝 곡률/폭 때문에 불가 → a_min을 손별 pad 폭의 1.5배로 클램프.
2. **엄지 DOF 차이** [M: URDF로 확인 필요]: shadow 엄지 5 DOF, sharpa·wuji·allegro·leap 엄지 DOF는 각기 다름(대략 4–5). 엄지 opposition의 **cross-palm**(K6–K10)은 CMC 외전·회전 둘 다 필요 → 손 별 K 도달 점수 먼저 측정 후 자세 목록에서 해당 손을 자동 마스크. 로봇 엄지 연구들은 Kapandji를 설계 평가에 사용하며 일부 손은 만점 보고 [V]; 점수는 "미리 기록한 자세 재생"으로 측정 [V] → 우리도 동일하게 손별 타깃 자세를 IK로 생성해 점수화.
3. **관절 한계·coupling**: 일부 손은 PIP–DIP 연동(underactuated/coupled)이라 DIP만 독립 제어 불가. IK에 coupling 제약 반영. 사람 taxonomy는 DIP 개별 굴곡을 가정하는 경우가 있어(예: 정확한 pad 접촉 각도) 로봇에서는 pad 법선 오차 허용치를 완화.
4. **소지 palm-arch / 손바닥 오목**: shadow 등은 소지 쪽 palm 관절이 있어 power wrap 접촉이 쉬운 반면 다른 손은 평판 palm → wrap 파지에서 접촉 면이 달라 **같은 aperture라도 접촉 지도가 다름**. palm 접촉은 "Pm 영역 존재 여부"로 마스크.
5. **dynamic primitive**: 접촉 변화(rolling/sliding) 동작은 kinematic IK만으로는 재현 불가(마찰·동역학 의존). 우리는 **kinematic 키프레임**만 만들고 "접촉이 유지되는 구간"을 라벨로 부여. 물리 검증이 필요하면 별도 시뮬(MuJoCo 등) 단계(이번 스펙 범위 밖).
6. **Santello 지적**: synergy가 grasp type과 일치하지 않음 [V] → 코덱 latent가 synergy 방향만 정렬하면 type 구별 정보를 잃는다 → type 라벨 기반 대조 지표(C4)를 평가에 포함.
7. **Taxonomy 자체 한계**: Feix는 정적·안정·단일손만 다룸 [V]; Liu 2014는 일상 행동에서 기존 분류에 없는 파지와 motion/force/stiffness 속성이 있음을 지적 [V] → 힘·stiffness는 kinematic 페어 데이터에 반영되지 않음. 본 스펙은 **위치 기능만** 정렬하며, 힘 정렬은 별도 과제로 명시.
8. **다른 연구의 처리 방식**:
   - Dexonomy: 손·type별 사람 템플릿 1개 → 대규모 합성 (손별 템플릿 주석이 필요) → 우리도 손별 "pad 프레임/접촉 주석 템플릿"을 1회 작성하는 접근이 현실적.
   - GenDexGrasp·CEDex: hand-agnostic contact map으로 손 간 전이.
   - CrossDex: 사람 손 공통 공간에 맵핑 후 각 로봇 kinematics로 retarget.
   - GraspGraphNet: URDF 기반 그래프로 morphology 조건.
   → 선택지: (A) 우리 방식 = 접촉/기능 정의 + IK, (B) contact-map 기반 생성. 페어 정렬 목적에는 (A)가 라벨(type) 부여에 유리.
9. **검증 필요 항목(미해결)**: (a) Feix 33종 정확한 VF/opp/thumb 표(원문 대조), (b) Kamakura 14종 명칭(1980 원문), (c) Elliott & Connolly 동작 명칭 정확성, (d) Kapandji 1–10 랜드마크 원문 기준, (e) 5손 엄지 DOF·관절 한계·palm 구조(URDF/데이터시트), (f) Allegro/LEAP 손가락 외전 범위, (g) 선행 연구에서 "taxonomy 라벨 + 손별 IK 페어"의 존재 여부.

---

## 8. 구현 순서 제안
1. URDF에서 손별 s, W, pad 프레임, 엄지 DOF·limit 추출 → 표(손×자세 feasibility) 자동 생성.
2. 코어 58 자세의 접촉 쌍·굴곡 prior를 YAML로 기록(§3 표 → 스키마).
3. 각 자세 Sobol 샘플 + IK → valid 필터 → 마스크 포함 페어 저장.
4. C1–C4 계산 보고서 → 빈 셀 보강.
5. 동적 primitive 14개: 키프레임 + 접촉 마스크.
6. (선택) 물리 시뮬 검증, 힘/stiffness 축 추가.

---

## 참고 링크 (전체)
- Cutkosky 1989: https://doi.org/10.1109/70.34763 ; https://biorobotics.harvard.edu/pubs/1990/human%20grasp%20choice.pdf
- Kamakura 1980: https://doi.org/10.5014/ajot.34.7.437 ; https://www.jstage.jst.go.jp/article/jjrm1964/15/2/15_2_65/_article
- Opposition space / virtual finger: https://www.cs.cmu.edu/afs/cs/user/sbk/www/thesis/opposition-space.html ; https://pmc.ncbi.nlm.nih.gov/articles/PMC4656720
- Feix 2016: https://ieeexplore.ieee.org/document/7243327 ; https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf
- Kapandji: https://en.wikipedia.org/wiki/Kapandji_score ; https://doi.org/10.1177/1753193419843837 ; 원논문 PMID 3963909
- Elliott & Connolly 1984 (이차): https://www.eng.yale.edu/grablab/pubs/Bullock_ICORR2011.pdf
- Exner / IHM: https://pmc.ncbi.nlm.nih.gov/articles/PMC4424153 ; https://scielo.org.za/pdf/sajot/v46n2/09.pdf
- Bullock 2013: https://www.eng.yale.edu/grablab/pubs/Bullock_TOH2013_1.pdf
- Liu 2014: https://www.homepages.ucl.ac.uk/~ucacyna/papers/TaxonomyEverydayGrasps.pdf ; https://ri.cmu.edu/?p=119868
- Stival 2019: https://pmc.ncbi.nlm.nih.gov/articles/PMC6377750/
- Santello 1998: https://pmc.ncbi.nlm.nih.gov/articles/PMC6793309 ; Weiss & Flanders 2004: https://www.ece.uvic.ca/~bctill/papers/mocap/Weiss_Flanders_2004.pdf
- Dexonomy: https://arxiv.org/abs/2504.18829
- DexGYS: https://arxiv.org/abs/2405.19291
- Cross-embodiment: https://arxiv.org/pdf/2410.02479 ; https://arxiv.org/abs/2603.16806 ; https://arxiv.org/pdf/2607.11031
- Robot Kapandji: https://arxiv.org/pdf/2201.10883 ; https://arxiv.org/pdf/2606.17418 ; https://arxiv.org/pdf/2606.14250 ; https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1496073/epub
