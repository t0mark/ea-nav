# 실제 로봇 자산 조사·수집 종합 (USD / URDF)

이 문서는 원래 네 개로 나뉘어 있던 기록을 하나로 합치고, 뒤에 다섯 번째 조사를 이어붙인 것이다.

1. **공개 USD 소스 조사** — 제조사·플랫폼이 직접 공개한 USD 확보 현황
2. **실제 로봇 URDF/USD 수집 목록** — 실제 수집·재컴파일·검증 이력과 카테고리별 목록
3. **Wheeled 로봇 누락 mesh 목록** — visual/collision mesh 확보 현황
4. **로봇 후보 목록** — 판단 없이 전수 나열한 브레인스토밍 목록
5. **로봇별 오픈소스 RL 보행 학습 리포지토리** — 워크스페이스 legged 로봇별 공개 학습 코드 링크 (2026-09-06)

---

## 1. 공개 USD 소스 조사

- 조사일: 2026-09-04 ~ 2026-09-05
- 목적: URDF -> USD 변환 없이, 제조사·플랫폼이 직접 공개한 USD를 로봇 타입(바퀴형/다족보행/휴머노이드)별로 확보
- 검증 방식: 검색 요약이 아니라 실제 파일 저장소(S3 버킷 리스팅, GitHub/HuggingFace 트리)를 직접 열어 `.usd` 파일 존재를 확인. 로그인 없이 바로 접근 가능한 것만 포함. 전부 Isaac Sim에 실제로 스폰해서(`tools/02_robot_spawn.py`) 화면에 정상적으로 나오는 것까지 확인함.
- 제외 대상: 로봇팔/그리퍼/손만 있는 매니퓰레이터 자산, 드론, RL 벤치마크용 가상 캐릭터(Ant, Cartpole 등), 실제 하드웨어가 아닌 예제용 자산(BalanceBot, Vehicle, Leatherback, Simple 등)

### 1.1 요약

| 카테고리 | 확인된 종류 수 | 주 출처 |
|---|---|---|
| 바퀴형 | 14 | NVIDIA Isaac Sim Nucleus 공식 자산 |
| 다족보행 | 17 | NVIDIA Isaac Sim Nucleus + Unitree + Deep Robotics 공식 저장소 |
| 휴머노이드 | 17 | NVIDIA Isaac Sim Nucleus + Unitree + LimX Dynamics 공식 저장소 |
| **합계** | **48** | |

### 1.2 바퀴형 (14종)

출처: NVIDIA Isaac Sim Nucleus 공식 자산 (`omniverse-content-production` S3 버킷, Isaac 4.5/5.1 버전 공통 확인)

| 로봇 | 제공처 | 경로(버킷 prefix 기준) |
|---|---|---|
| Create3 | iRobot | `Robots/iRobot/create_3.usd` |
| Turtlebot3 Burger | Turtlebot | `Robots/Turtlebot/turtlebot3_burger.usd` |
| Jetbot | NVIDIA | `Robots/NVIDIA/Jetbot/jetbot.usd` |
| AWS RoboMaker Jetbot | NVIDIA/AWS | `Robots/NVIDIA/Jetbot/aws_robomaker_jetbot.usd` |
| Carter v1 | NVIDIA | `Robots/NVIDIA/Carter/carter_v1...usd` |
| Nova Carter (+Dev Kit) | NVIDIA | `Robots/NVIDIA/NovaCarter/`, `NovaCarterDevKit/` |
| Kaya | NVIDIA | `Robots/NVIDIA/Kaya/kaya.usd` |
| Jackal | Clearpath | `Robots/Clearpath/Jackal/jackal.usd` |
| Dingo | Clearpath | `Robots/Clearpath/Dingo/dingo.usd` |
| Ridgeback | Clearpath | `Robots/Clearpath/RidgebackFranka/`, `RidgebackUr/` (베이스+암 결합체로만 존재) |
| iwhub | Idealworks | `Robots/Idealworks/iwhub/iw_hub.usd` |
| Evobot | Fraunhofer | `Robots/Fraunhofer/Evobot/evobot.usd` |
| O3dyn | Fraunhofer | `Robots/Fraunhofer/O3dyn/o3dyn.usd` |
| Limo | AgilexRobotics | `Robots/AgilexRobotics/limo/limo.usd` |

**참고(공식 USD 아님, 커뮤니티 포팅)**: PAL Robotics TIAGo/TIAGo++ Omni — [AIS-Bonn/tiago_isaac](https://github.com/AIS-Bonn/tiago_isaac) 저장소에서 메카넘 구동 포함 포팅 제공.

**확인 결과 없는 것**: Clearpath Husky(Nucleus에 Jackal/Dingo/Ridgeback만 있고 Husky 폴더 자체가 없음), PAL Talos.

### 1.3 다족보행 (17종)

| 로봇 | 제공처 | 출처 |
|---|---|---|
| A1, Go1, Go2, B2, Aliengo, Laikago | Unitree | Isaac Sim Nucleus `Robots/Unitree/{모델}/` |
| Go2W (바퀴+다리 하이브리드) | Unitree | 공식 GitHub/HuggingFace [`unitreerobotics/unitree_model`](https://huggingface.co/datasets/unitreerobotics/unitree_model) `Go2W/usd/` |
| ANYmal-B, ANYmal-C, ANYmal-D | ANYbotics | Isaac Sim Nucleus `Robots/ANYbotics/anymal_{b,c,d}/` |
| Spot (+with arm) | Boston Dynamics | Isaac Sim Nucleus `Robots/BostonDynamics/spot/` |
| Lite3, X30, M20, M20S, M20_Piper(+로봇팔), DR02(standard) | Deep Robotics | 공식 GitHub [`DeepRoboticsLab/deep_robotics_model`](https://github.com/DeepRoboticsLab/deep_robotics_model) |

**확인 결과 없는 것**: Ghost Robotics(Vision 60 등), Xiaomi CyberDog, LimX Dynamics 사족(W1/P1/CL-1, 공개 저장소에서 USD 확인 못함 - 휴머노이드/TRON2 계열만 USD 있음).

**제외함**: `Robots/NTNU/ARL-Robot-1/` — 실제 스폰 검증 과정에서 확인해보니 다족보행 로봇이 아니라 프로펠러 4개가 고정 관절(PhysicsFixedJoint)로 붙은 드론(쿼드콥터)이었음. 회전/직동 관절이 없어 관절 기반 로봇 검증에도 안 맞고, plan.md의 타겟 하드웨어(바퀴형/다족보행/휴머노이드)에도 해당하지 않아 목록에서 제외.

### 1.4 휴머노이드 (17종)

| 로봇 | 제공처 | 출처 |
|---|---|---|
| H1 (+with hand) | Unitree | Isaac Sim Nucleus `Robots/Unitree/H1/` |
| G1 (+with hand, 23dof) | Unitree | Isaac Sim Nucleus `Robots/Unitree/G1/`, `G1_23dof/` |
| H1-2, H2 | Unitree | 공식 GitHub/HuggingFace `unitree_model` `H1-2/`, `H2/` (H2는 usd 파일 크기가 1.45KB로 매우 작아 실사용 전 내용 확인 필요) |
| GR-1 (T1, T2) | Fourier Intelligence | Isaac Sim Nucleus `Robots/FourierIntelligence/GR-1/` |
| Digit v4 | Agility Robotics | Isaac Sim Nucleus `Robots/Agility/Digit/digit_v4.usd` |
| Neo | 1X | Isaac Sim Nucleus(5.1) `Robots/1X/Neo/Neo.usd` |
| STAR1 | RobotEra | Isaac Sim Nucleus(5.1) `Robots/RobotEra/STAR1/star1.usd` |
| Phoenix | SanctuaryAI | Isaac Sim Nucleus(5.1) `Robots/SanctuaryAI/Phoenix/phoenix.usd` |
| PX5 | XiaoPeng | Isaac Sim Nucleus(5.1) `Robots/XiaoPeng/PX5/px5.usd` |
| A2D | Agibot | Isaac Sim Nucleus(5.1) `Robots/Agibot/A2D/A2D.usd` |
| T1 (locomotion) | Booster Robotics | Isaac Sim Nucleus(5.1) `Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd` |
| Valkyrie | IHMC Robotics | Isaac Sim Nucleus(5.1) `Robots/Ihmcrobotics/Valkyrie/valkyrie.usd` |
| HU_D03, HU_D04(+그리퍼) | LimX Dynamics | 공식 GitHub [`limxdynamics/humanoid-description`](https://github.com/limxdynamics/humanoid-description) |
| TRON2A SF(하반신, 이족보행) | LimX Dynamics | 공식 GitHub [`limxdynamics/tron2-robot-description`](https://github.com/limxdynamics/tron2-robot-description) — 8개 변형 중 대표로 뽑음, 팔·상체 없이 다리만 있는 로코모션 베이스 |
| TRON2A WF(하반신, 바퀴-다리 하이브리드) | LimX Dynamics | 위와 동일 저장소 — 발 대신 바퀴가 달린 변형 |

**확인 결과 없는 것**: Tesla Optimus, PAL Talos, Ghost Robotics 계열, Booster K1/T2(T1만 확인), Unitree R1/H2 Plus, Galbot 휴머노이드 라인(`galbot_s1_description`은 지오메트리 페이로드만 있고 완성된 진입점 USD가 없어 제외), Agibot X2/X2Ultra(공개 저장소는 URDF만 확인됨).

**뒤늦게 제외함**: Tien Kung(XHumanoid) — 이 조사 당시엔 스폰 검증(화면에 정상적으로 나옴)까지만 확인했는데, 이후 legged RL 학습 파이프라인 전수 검증(2026-09-06)에서 관절 트리 루트가 3개(몸통+양손이 서로 미연결)인 결함이 발견돼 usd·urdf를 전부 삭제했다 - 자세한 내용은 아래 «2.2 전수 재검증 및 정리»의 "원본 자체의 한계라 삭제한 것" 참고. 스폰 검증만으로는 킨매틱 트리 연결성까지 보장하지 못한다는 사례.

### 1.5 방법론 메모

- NVIDIA Isaac Sim 자산은 `https://omniverse-content-production.s3-us-west-2.amazonaws.com/`가 로그인 없이 공개 리스팅되는 S3 버킷이라, `?list-type=2&prefix=...&delimiter=/` 쿼리로 폴더 구조와 실제 파일 존재 여부를 직접 확인할 수 있었다. 버전마다(4.5 vs 5.1) 포함된 로봇이 달라서 최소 두 버전을 함께 확인해야 누락이 없다.
- Unitree는 Isaac Nucleus에 없는 최신·변형 모델(H1-2, H2, Go2W)을 자체 GitHub/HuggingFace 저장소(`unitreerobotics/unitree_model`)에 별도로 공개하고 있어, 두 소스를 모두 확인해야 한다.
- Deep Robotics·LimX Dynamics처럼 회사가 자체 GitHub에 `usd/` 폴더를 직접 커밋해 공개하는 경우가 있다 - Isaac Nucleus에 없다고 "USD 공개 안 됨"으로 단정하면 안 되고, 회사 GitHub 계정을 따로 확인해야 한다(초기 조사에서 Deep Robotics·LimX를 "확인 결과 없음"으로 잘못 적었던 이유).
- "USD 파일이 저장소에 존재한다"와 "Isaac Sim에서 바로 스폰해서 정상 작동한다"는 별개 문제다. 특히 크기가 비정상적으로 작은 파일(H2의 1.45KB 등)은 참조용 스텁일 가능성이 있어 실제 사용 전 열어서 확인이 필요하다.
- **텍스처(색) 유무는 출처에 따라 갈린다.** NVIDIA가 직접 공들인 "플래그십" Nucleus 자산(Aliengo, Spot, ANYmal, GR-1, Valkyrie, Nova Carter 등)은 사진 같은 PBR 텍스처가 있지만, 회사가 CAD/STL을 자동 변환 도구로 그대로 내보낸 경우(Deep Robotics, LimX, Turtlebot3, iwhub 등)는 재질이 `diffuse_color_constant = (1,1,1)`(흰색) 하나로 통일돼 있다 - STL 자체가 색 정보를 담지 못하는 포맷이라 변환 과정에서 색이 통째로 유실된 것으로, 파이프라인 버그가 아니라 원본 데이터의 한계다. 형상·관절 검증에는 영향 없음.

---

## 2. 실제 로봇 URDF/USD 수집 목록

- 최종 갱신 : 2026-09-06
- 총 123종. 폴더 구조: `wheeled/{diff(37),ackermann(2),omni(12)}`, `legged/{multi-legged(33),humanoid(39)}`
- 파일 구조 : 로봇당 **urdf 1개** (`{category}/.../{company}_{robot}.urdf`), USD가 원본 저장소에 실제로 있는 로봇만 **usd 1개** (`data/sim/usd/real_robot/legged/{company}_{robot}.usd`)

### 2.1 자크로 재컴파일 대조 검증 (2026-09-04)

이전 연결성 검사(2026-09-03)는 완성된 URDF 자체의 구조만 봤을 뿐, "원본을 지금 다시 xacro로 돌리면 저장된 파일과 똑같은 결과가 나오는가"는 확인한 적이 없었다. 그래서 124종 전부를 원본 저장소에서 **클린 상태로 새로 받아 xacro로 재컴파일**한 뒤, 저장된 최종 URDF와 대조했다.

**비교 방식**: 텍스트 diff가 아니라 각 URDF의 링크 이름 집합·조인트 이름 집합을 뽑아 두 집합이 완전히 같은지 비교했다(자동 생성 주석의 경로, 속성 순서, 부동소수점 표현 등 의미 없는 차이를 걸러내기 위함). 카테고리별로 4묶음(wheeled/diff 36 + pal_tiago 별도, wheeled/ackermann+omni 14, legged/multi-legged 33, legged/humanoid 39 + pal_talos 별도)으로 나눠 순차 진행했다. 결과: **124/124 최종 일치** (아래 3건은 대조 과정에서 실제 오류를 발견해 수정한 뒤 일치).

**대조 중 발견된 실제 데이터 오류 (수정 완료)**
| 로봇 | 문제 | 원인 | 조치 |
|---|---|---|---|
| mobilerobots_pioneer_lx | 엉뚱한 로봇 데이터가 저장돼 있었음 | 예전에 pioneer 계열 저장소 하나를 로봇별로 수동 분리하던 중 실수로 pioneer3at의 내용을 pioneer_lx 파일에 넣음(`robot name="pioneer3at"`) | 올바른 pioneer-lx 소스로 재컴파일해 교체(4링크: base_link, r_wheel, l_wheel, deck) |
| pal_tiago | 베이스·바퀴·센서가 통째로 빠짐(31링크) | 검증 스크립트의 include 정리 로직이 `$(find ${base_type}_description)`처럼 변수가 섞인 include를 "지역 패키지 아님"으로 오판해 삭제 — 최신 tiago_robot 저장소는 실제로는 base_type 인자로 pmb2_description을 그대로 합치는 단일 진입 파일(`tiago.urdf.xacro`)을 갖고 있었음 | 스크립트 버그 수정(`${`가 섞인 find 인자는 정적 판단하지 않고 통과) 후 pmb2_description·pal_urdf_utils 패키지를 합쳐 재컴파일, 완전판(50링크, 단일 루트, 고아 없음)으로 교체 |
| pal_talos | IMU·RGBD 카메라·손목 F/T 센서 링크 11개 누락(41링크) | 원인 특정 못함(그리퍼 고아 링크 제거 작업 중 함께 잘려나간 것으로 추정) | 기본 인자로 재컴파일한 완전판(52링크, 단일 루트, 고아 없음)으로 교체 |

**대조 과정에서 함께 고친 검증 스크립트 자체의 버그** (저장된 데이터에는 영향 없었던 것)
- `candidate_entries`(진입 파일 자동 탐지)가 대문자 확장자(`.URDF`)를 못 잡던 것, macOS 압축 해제 잔여물(`__MACOSX/._*.urdf`, 바이너리라 파싱 시 깨짐)을 후보로 잡던 것 → 대소문자 무시 + `__MACOSX`/점파일 제외로 수정 (xhumanoid_tiangong, poppy 계열에서 발견)
- `old_property` 보정 로직이 파일 앞부분 몇 줄만 보고 `xmlns:xacro` 존재 여부를 판단해 일부 파일에서 선언을 못 넣던 것 → 파일 전체 검사로 수정 (segway_rmp440le에서 발견)
- softbank_nao·softbank_romeo·softbank_pepper: 저장소 안에 버전별 사전 생성(pre-generated) 완성 URDF가 따로 있는데 자동 탐지가 애매하게 점수를 매겨 다른 조각 파일을 고르던 문제 → 해당 3종은 진입 파일을 사전 생성 URDF 경로로 직접 지정. 실제 저장 파일은 링크·조인트 집합이 정확히 일치해 데이터 자체는 처음부터 정상이었음(nao는 naoV33/V40 두 버전이 동일 구조로 확인됨)

### 2.2 전수 재검증 및 정리 (2026-09-03)

킨매틱 트리 연결성(모든 링크가 루트에서 도달 가능한지)과 카테고리별 최소 구조를 자동 검사한 뒤, 걸린 건 전부 원본까지 다시 파서 고쳤다. 고쳐도 원본 자체의 한계로 완성본이 안 나오는 것들은 삭제했다.

**진짜로 깨져 있던 것 (수정 완료)**
| 로봇 | 문제 | 원인 | 조치 |
|---|---|---|---|
| clearpath_boxer | base_link 없음 | 이전 처리 과정 오염(원인 특정 못함) | 클린 재다운로드로 해결 (5→27링크) |
| neobotix_mmo500 | 팔 전체가 분리됨 | 원본 저장소에서 팔이 붙는 `cabinet_link`을 만드는 매크로 호출이 주석 처리되어 있었음 | 주석 해제 (41링크, 완전 연결) |
| neobotix_mpo700 | 바퀴/캐스터 조인트가 전부 fixed | 원본 매크로 정의 자체의 오류(axis·limit·velocity가 있는데 type만 fixed) | continuous로 수정 |
| iit_coman, iit_walkman | 팔다리가 통째로 빠짐(9링크) | `package.xml`이 없는 구식(pre-catkin) 저장소라 로컬 패키지 판별에 실패해 관련 include를 전부 제거해버림 | 판별 로직에 manifest.xml·CMakeLists.txt 디렉토리 인식 추가 (68, 59링크) |
| pal_talos, pal_reemc | 그리퍼 하위 부품이 고아 링크로 남음 | 손 매크로 호출은 지웠는데 하위 부품 정의는 남아있었음 | 고아 링크 일괄 제거 (pal_talos는 이 과정에서 센서 링크까지 함께 잘려나간 게 2026-09-04 재검증에서 추가로 드러나 완전판으로 재교체됨 — 위 섹션 참고) |
| poppyproject_poppy_ergo_jr | 루트 2개 | 말단 조인트에 `<parent>` 태그 누락(원본 결함) | section_5에 연결 |
| pal_ari, pal_tiago | 베이스(바퀴)가 통째로 빠짐 | 잘못된 진입 파일 사용(ARI) / TIAGo는 당시 베이스와 상체를 합치는 진입 파일이 없다고 판단해 직접 작성했으나 31링크로 불완전했음(진짜 원인은 2026-09-04 재검증에서 밝혀짐 — 위 섹션 참고) | ARI는 올바른 진입 파일로 재처리(39링크), TIAGo는 2026-09-04에 50링크 완전판으로 재교체 |
| clearpath_turtlebot4 | 바퀴 없음(21링크) | Create3 베이스를 가져오는 include가 외부 패키지로 오인되어 제거됨 | irobot_create_description/control 패키지를 합쳐서 재처리 (50링크, 완전 연결) |
| agilex_limo | 팔 관절만 남고 베이스가 없음 | 잘못된 진입 파일 사용 | limo_four_diff.xacro(4륜 디퍼렌셜 모드)로 재처리 (11링크) |

**처음엔 의심했지만 확인해보니 정상인 것**
- `metralabs_scitos_g5`: 바퀴 조인트가 없는 게 아니라, 이 로봇의 원본 메시 자체가 바퀴를 별도 링크로 분리하지 않고 통짜 바디로 모델링되어 있었음 (diff로 재분류)
- `berkeley_humanoid`(다리만 있는 게 정상, 실제 하드웨어도 하체 전용), `westwood_bruce`, `poppyproject_poppy_torso`(원래 상체 전용) — 링크 수는 적지만 연결성 문제 없음

**원본 자체의 한계라 삭제한 것 (5종)**
- `nasa_valkyrie`: 2015년식 NASA 저장소의 매크로 스코프(파일 include 시점 vs 매크로 파라미터 시점)가 근본적으로 꼬여있어 상반신 프레임(12링크)까지밖에 못 살렸다. 팔·다리·손가락마다 반복되는 동일 아키텍처 결함을 전부 고쳐야 완전해지는데 실익 대비 시간이 너무 들어 삭제
- `festo_robotino`: 저장소 자체에 base_link 하나만 정의돼 있고 바퀴 지오메트리가 전혀 없음
- `mobilerobots_powerbot`: 저장소 자체가 "정확한 형상 아님, 블록으로만 표현" 이라고 명시된 최소 placeholder 모델이라 바퀴 없음
- `agilex_bunker`: 바퀴가 아니라 궤도(트랙) 차량이라 애초에 diff/ackermann/omni 어디에도 안 맞음
- `xhumanoid_tiangong`: legged RL 학습 파이프라인 전수 검증(2026-09-06) 중 발견 - 실제 사용하던 usd(Isaac Sim Nucleus 배포본, `data/sim/usd/real_robot/legged/humanoid/xhumanoid_tiengong.usd`)의 관절 트리 루트가 3개(`pelvis`, 좌/우 손 base_link)로 나뉘어 있어 양손이 몸통과 물리적으로 연결되지 않은 상태였다. 스폰 자체는 되고 화면에도 정상적으로 나와 위 «1. 공개 USD 소스 조사»의 시각 검증은 통과했었지만, 관절 트리 연결성까지는 그때 확인하지 못했다. usd·urdf(및 대응 mesh/texture 에셋)를 전부 삭제

**중요**: 위 5종 중 `xhumanoid_tiangong`은 다른 4종과 달리 usd까지 이미 있었는데도(위 «1. 공개 USD 소스 조사»의 49종 확인 목록에 포함) 뒤늦게 발견됐다 - "스폰돼서 화면에 나온다"는 킨매틱 트리가 정상이라는 뜻이 아니므로, 앞으로 humanoid USD를 새로 들일 때는 관절 트리 루트가 하나인지도 같이 확인해야 한다.

**제외(수집 자체를 안 함)**
- `pollen_reachy2`: 팔 관절(어깨/팔꿈치)이 비공개 액추에이터 패키지(`orbita2d_description`/`orbita3d_description`)에 종속돼 있어 완성본을 만들 수 없었음

### 2.3 만든 방식

1. 회사 단위로 실제 GitHub URDF 공개 여부를 먼저 조사(아래 «4. 로봇 후보 목록»)한 뒤, 검증된 저장소만 로봇별로 받았다 (같은 저장소를 쓰는 로봇은 한 번만 다운로드해 재사용).
2. 각 로봇의 xacro 진입 파일을 찾아 **xacro로 완전히 전개**해서 include/매크로가 여러 파일로 쪼개져 있던 것을 로봇당 단일 URDF로 합쳤다. 처리 컨테이너: `airlab_hw_eatrav_utils` (xacro 2.1.1 + lxml).
   - ROS2 xacro가 쓰는 `$(find 패키지)` 치환은 `ament_index_python`이 필요한데 정식 ROS 환경이 없어서, 저장소 내 `package.xml`/`manifest.xml`을 스캔해 패키지명→경로를 매핑하는 최소 셈(stub) 모듈을 만들어 대체했다. 매니페스트가 아예 없는 구식 저장소는 `CMakeLists.txt`가 있는 디렉토리명을 패키지명으로 취급한다.
   - 로컬에 없는 외부 패키지를 참조하는 `<xacro:include>`는 안전하게 제거했고, 그 결과 "정의되지 않은 매크로" 에러가 나면 해당 매크로 **호출부**를 찾아 반복 제거하는 자동 루프를 돌렸다 (카메라·라이다·IMU 같은 부속 센서 매크로가 대부분 — core 로봇 몸체에는 영향 없음).
   - 베이스/팔 등 핵심 몸체가 별도의 소형 공개 패키지(pmb2_description, irobot_create_description 등)에 있는 경우 그 저장소도 받아서 합쳤다.
   - 일부 저장소는 결함(선언 안 된 xacro:arg, 옛날 방식 `<property>`/`<macro>` 태그에 `xacro:` 접두어 누락, 주석 처리된 매크로 호출, 잘못된 조인트 타입, 존재하지 않는 상수 참조 등)이 있어 직접 고쳤다.
   - 순수 `.urdf`(xacro 아님) 형태로 이미 완성돼 있던 로봇은 xacro 처리 없이 그대로 받았다.
3. 완성 후 모든 URDF에 대해 킨매틱 트리 연결성(고아 링크·다중 루트)을 검사해 재검증했다.
4. USD가 저장소에 실제로 커밋되어 있던 Deep Robotics 4종(Lite3, X30, M20, DR02)은 `_base/_robot/_sensor/_physics`로 쪼개진 레이어 구성을 **usd-core(pxr)의 `Stage.Flatten()`으로 평탄화**해서 로봇당 완성본 usd 파일 하나로 합쳤다 (원본 레이어 파일들은 결과물에 포함하지 않음).
5. mesh(stl/dae/obj), launch, rviz, package.xml 등은 처음부터 받지 않았다.

### 2.4 알아둘 점

- 라이선스 원문은 로봇 폴더에서는 제거했다 — 재배포 전 원 저장소 재확인 필요 (특히 Fetch는 CC-BY-NC-SA, GR 시리즈는 GPL-3.0, Segway RMP440LE는 "Segway 제품 전용" 제한 라이선스).
- xacro 처리 과정에서 일부 로봇은 센서 부착물(카메라 마운트, 특정 라이다/IMU 매크로 등)이 "정의되지 않아서" 자동으로 빠졌다 — **이동 몸체(링크·조인트) 자체는 전부 온전**하지만, 정확한 센서 페이로드 구성까지 필요하면 원 저장소를 다시 확인해야 한다.
- Jackal/TurtleBot3/Create3/H1/G1의 USD는 Isaac Sim 자산 서버·HuggingFace 경유로만 공개돼 있고 GitHub 저장소에는 없어서 이 환경(GitHub만 접근 가능)에서는 만들지 못했다.
- DR02는 저장소에 `standard`/`pro` 두 트림이 있는데, urdf·usd 모두 `standard(DR02-std)` 트림 하나만 남겼다.

### 2.5 Wheeled (51) — diff(37)/ackermann(2)/omni(12) 폴더로 분류

**diff**: limo, ranger_mini, scout, tracer, astribot_s1, boxer, dingo, husky, jackal, turtlebot4, warthog, fetch, stretch, panther, create3, scitos_g5, mrp2, mir100, pioneer3at, pioneer3dx, pioneer_lx, seekurjr, botvac, mp500, ld60, otto_1500, otto_600, ari, tiago, turtlebot3, rb1_base, rbtheron, rbwatcher, summit_xl, rmp440le, hsr, kobuki

**ackermann**: hunter, f1tenth

**omni**: ridgeback, galbot_s1, rosbot, kmriiwa, youbot, mmo500, mpo500, mpo700, rbkairos, rbvogui, pepper, pr2

### 2.6 Legged (33) — multi-legged/humanoid 폴더로 분류

**multi-legged**: anymal_b, anymal_c, anymal_d, spot, dr02(usd 포함), lite3(usd 포함), m20(usd 포함), x30(usd 포함), spirit40, vision60, barkour, jethexa, puppypi, hyq, wl_p311d, mini_pupper, mini_cheetah, solo, bittle, rbq10, doggo, pupper_v3, phantomx_hexapod, wxmark4hexapod, a1, aliengo, b1, b2, b2w, go1, go2, laikago, minitaur

이 중 6족(헥사포드)은 phantomx_hexapod, trossen_wxmark4hexapod, hiwonder_jethexa 3종, 나머지는 전부 4족.

### 2.7 Humanoid (39)

x2ultra, cassie, digit, berkeley_humanoid, nimbro_op2, k1, t1, atlas, surena_v, pm01, gr1t2, gr2, gr3, coman, ergocub, icub, walkman, inmoov, hubo, kuavo, hud04, magicbotz1, qinglong, reemc, talos, poppy, poppy_ergo_jr, poppy_torso, star1, op3, thormang3, nao, romeo, g1, h1, h2, r1, draco3, bruce

tiangong은 관절 트리 결함으로 삭제됨(위 «2.2 전수 재검증 및 정리»의 "원본 자체의 한계라 삭제한 것" 참고).

각 robot_id의 회사명, 출처 저장소, 라이선스는 파일명(`{company}_{robot_id}.urdf`)과 아래 «4. 로봇 후보 목록»을 참고.

---

## 3. Wheeled 로봇 누락 mesh 목록

- 조사일: 2026-09-04
- 조사 대상: data/urdf/real_robot/wheeled/ 전체 (mesh를 참조하는 로봇 기준)
- 누락 있는 로봇: 44대 / 총 누락 참조: 371개
- collision 태그 mesh는 전수 확인 결과 100% 확보됨 (물리 시뮬레이션에는 영향 없음)
- 아래 [visual]만 있는 항목은 화면에 안 보이는 것 외에 기능적 영향 없음, [collision]이 섞여 있으면 우선 확인 필요

### f1tenth_f1tenth (ackermann, 5개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/ackermann/f1tenth_f1tenth`
  - [visual] package://f1tenth-sim/urdf/meshes/chassis.stl
  - [visual] package://f1tenth-sim/urdf/meshes/hinge.stl
  - [visual] package://f1tenth-sim/urdf/meshes/hokuyo.stl
  - [visual] package://f1tenth-sim/urdf/meshes/left_wheel.stl
  - [visual] package://f1tenth-sim/urdf/meshes/right_wheel.stl

### agilex_limo (diff, 2개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/agilex_limo`
  - [visual] package://limo_description/meshes/limo_base.dae
  - [visual] package://limo_description/meshes/limo_wheel.dae

### agilex_tracer (diff, 1개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/agilex_tracer`
  - [visual] package://tracer_description/meshes/castor_joint.dae

### astribot_s1 (diff, 22개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/astribot_s1`
  - [visual] meshes/dae/astribot_head_link_2.dae
  - [visual] meshes/dae/astribot_torso_base.dae
  - [visual] meshes/dae/astribot_torso_link_1.dae
  - [visual] meshes/dae/astribot_torso_link_2.dae
  - [visual] meshes/dae/astribot_torso_link_4.dae
  - [visual] meshes/s1_arm/astribot_arm_left_base_link.STL
  - [visual] meshes/s1_arm/astribot_arm_left_link_2.STL
  - [visual] meshes/s1_arm/astribot_arm_link_1.STL
  - [visual] meshes/s1_arm/astribot_arm_link_3.STL
  - [visual] meshes/s1_arm/astribot_arm_link_4.STL
  - [visual] meshes/s1_arm/astribot_arm_link_5.STL
  - [visual] meshes/s1_arm/astribot_arm_link_6.STL
  - [visual] meshes/s1_arm/astribot_arm_link_7.STL
  - [visual] meshes/s1_arm/astribot_arm_right_base_link.STL
  - [visual] meshes/s1_arm/astribot_arm_right_link_2.STL
  - [visual] meshes/s1_head/astribot_head_base_link.STL
  - [visual] meshes/s1_head/astribot_head_link_1.STL
  - [visual] meshes/s1_torso/astribot_torso_link_3.STL
  - [visual] s1_torso/wheel_LF_Link.STL
  - [visual] s1_torso/wheel_LR_Link.STL
  - [visual] s1_torso/wheel_RF_Link.STL
  - [visual] s1_torso/wheel_RR_Link.STL

### clearpath_boxer (diff, 4개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/clearpath_boxer`
  - [visual] package://boxer_description/meshes/boxer24_no_sensors.stl
  - [visual] package://boxer_description/meshes/hokuyo_uam_05lp.stl
  - [visual] package://boxer_description/meshes/rear_sensor.stl
  - [visual] package://boxer_description/meshes/vecow_evs2000.stl

### clearpath_dingo (diff, 2개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/clearpath_dingo`
  - [visual] package://clearpath_platform_description/meshes/dd100/chassis.dae
  - [visual] package://clearpath_platform_description/meshes/dd100/wheels/indoor.stl

### clearpath_husky (diff, 6개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/clearpath_husky`
  - [visual] package://husky_description/meshes/base_link.dae
  - [visual] package://husky_description/meshes/bumper.dae
  - [visual] package://husky_description/meshes/top_chassis.dae
  - [visual] package://husky_description/meshes/top_plate.dae
  - [visual] package://husky_description/meshes/user_rail.dae
  - [visual] package://husky_description/meshes/wheel.dae

### clearpath_jackal (diff, 3개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/clearpath_jackal`
  - [visual] package://jackal_description/meshes/jackal-base.stl
  - [visual] package://jackal_description/meshes/jackal-fender.stl
  - [visual] package://jackal_description/meshes/jackal-wheel.stl

### clearpath_turtlebot4 (diff, 9개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/clearpath_turtlebot4`
  - [visual] package://irobot_create_description/meshes/body_visual.dae
  - [visual] package://irobot_create_description/meshes/bumper_visual.dae
  - [visual] package://turtlebot4_description/meshes/camera_bracket.dae
  - [visual] package://turtlebot4_description/meshes/oakd_pro.dae
  - [visual] package://turtlebot4_description/meshes/rplidar.dae
  - [visual] package://turtlebot4_description/meshes/shell.dae
  - [visual] package://turtlebot4_description/meshes/tower_sensor_plate.dae
  - [visual] package://turtlebot4_description/meshes/tower_standoff.dae
  - [visual] package://turtlebot4_description/meshes/weight_block.dae

### clearpath_warthog (diff, 7개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/clearpath_warthog`
  - [visual] package://clearpath_platform_description/meshes/w200/chassis.stl
  - [visual] package://clearpath_platform_description/meshes/w200/diff-link.stl
  - [visual] package://clearpath_platform_description/meshes/w200/e-stop.stl
  - [visual] package://clearpath_platform_description/meshes/w200/light.stl
  - [visual] package://clearpath_platform_description/meshes/w200/rocker.stl
  - [visual] package://clearpath_platform_description/meshes/w200/susp-link.stl
  - [visual] package://clearpath_platform_description/meshes/w200/wheels/outdoor.stl

### fetchrobotics_fetch (diff, 17개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/fetchrobotics_fetch`
  - [visual] package://fetch_description/meshes/base_link.dae
  - [visual] package://fetch_description/meshes/bellows_link.STL
  - [visual] package://fetch_description/meshes/elbow_flex_link.dae
  - [visual] package://fetch_description/meshes/estop_link.dae
  - [visual] package://fetch_description/meshes/forearm_roll_link.dae
  - [visual] package://fetch_description/meshes/gripper_link.dae
  - [visual] package://fetch_description/meshes/head_pan_link.dae
  - [visual] package://fetch_description/meshes/head_tilt_link.dae
  - [visual] package://fetch_description/meshes/l_wheel_link.STL
  - [visual] package://fetch_description/meshes/r_wheel_link.STL
  - [visual] package://fetch_description/meshes/shoulder_lift_link.dae
  - [visual] package://fetch_description/meshes/shoulder_pan_link.dae
  - [visual] package://fetch_description/meshes/torso_fixed_link.dae
  - [visual] package://fetch_description/meshes/torso_lift_link.dae
  - [visual] package://fetch_description/meshes/upperarm_roll_link.dae
  - [visual] package://fetch_description/meshes/wrist_flex_link.dae
  - [visual] package://fetch_description/meshes/wrist_roll_link.dae

### hellorobot_stretch (diff, 22개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/hellorobot_stretch`
  - [visual] ./meshes/base_link.STL
  - [visual] ./meshes/d435.dae
  - [visual] ./meshes/laser.STL
  - [visual] ./meshes/link_DW3_tablet_12in.STL
  - [visual] ./meshes/link_DW3_wrist_pitch.STL
  - [visual] ./meshes/link_DW3_wrist_roll.STL
  - [visual] ./meshes/link_DW3_wrist_yaw_bottom.STL
  - [visual] ./meshes/link_SE3_head_nav_cam.STL
  - [visual] ./meshes/link_arm_l0.STL
  - [visual] ./meshes/link_arm_l1.STL
  - [visual] ./meshes/link_arm_l2.STL
  - [visual] ./meshes/link_arm_l3.STL
  - [visual] ./meshes/link_arm_l4.STL
  - [visual] ./meshes/link_head.STL
  - [visual] ./meshes/link_head_pan.STL
  - [visual] ./meshes/link_head_tilt.STL
  - [visual] ./meshes/link_left_wheel.STL
  - [visual] ./meshes/link_lift.STL
  - [visual] ./meshes/link_mast.STL
  - [visual] ./meshes/link_right_wheel.STL
  - [visual] ./meshes/link_wrist_yaw.STL
  - [visual] ./meshes/omni_wheel_m.STL

### husarion_panther (diff, 5개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/husarion_panther`
  - [visual] package://husarion_ugv_description/meshes/WH01/fl_wheel.dae
  - [visual] package://husarion_ugv_description/meshes/WH01/fr_wheel.dae
  - [visual] package://husarion_ugv_description/meshes/WH01/rl_wheel.dae
  - [visual] package://husarion_ugv_description/meshes/WH01/rr_wheel.dae
  - [visual] package://husarion_ugv_description/meshes/panther/base.dae

### irobot_create3 (diff, 2개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/irobot_create3`
  - [visual] package://irobot_create_description/meshes/body_visual.dae
  - [visual] package://irobot_create_description/meshes/bumper_visual.dae

### metralabs_scitos_g5 (diff, 7개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/metralabs_scitos_g5`
  - [visual] package://scitos_description/meshes/head_frame.dae
  - [visual] package://scitos_description/meshes/pantilt0.dae
  - [visual] package://scitos_description/meshes/pantilt1.dae
  - [visual] package://scitos_description/meshes/pantilt2.dae
  - [visual] package://scitos_description/meshes/pantilt3.dae
  - [visual] package://scitos_description/meshes/scitos_1.dae
  - [visual] package://scitos_description/meshes/xtion.dae

### milvus_mrp2 (diff, 6개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/milvus_mrp2`
  - [visual] package://mrp2_description/meshes/caster_base.stl
  - [visual] package://mrp2_description/meshes/caster_roller.stl
  - [visual] package://mrp2_description/meshes/left_wheel.stl
  - [visual] package://mrp2_description/meshes/right_wheel.stl
  - [visual] package://mrp2_description/meshes/sonar.stl
  - [visual] package://mrp2_description/meshes/utm_30lx.stl

### mir_mir100 (diff, 2개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/mir_mir100`
  - [visual] package://mir_description/meshes/visual/caster_wheel_base.stl
  - [visual] package://mir_description/meshes/visual/mir_100_base.stl

### mobilerobots_pioneer3at (diff, 4개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/mobilerobots_pioneer3at`
  - [visual] package://amr_robots_description/meshes/p3at_meshes/back_sonar.stl
  - [visual] package://amr_robots_description/meshes/p3at_meshes/front_sonar.stl
  - [visual] package://amr_robots_description/meshes/p3at_meshes/top.stl
  - [visual] package://amr_robots_description/meshes/p3at_meshes/wheel.stl

### mobilerobots_pioneer3dx (diff, 11개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/mobilerobots_pioneer3dx`
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/back_sonar.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/caster_hubcap.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/caster_swivel.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/caster_wheel.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/chassis.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/front_sonar.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/left_hubcap.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/left_wheel.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/right_hubcap.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/right_wheel.stl
  - [visual] package://amr_robots_description/meshes/p3dx_meshes/top.stl

### mobilerobots_pioneer_lx (diff, 4개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/mobilerobots_pioneer_lx`
  - [visual] package://amr_robots_description/meshes/p3at_meshes/back_sonar.stl
  - [visual] package://amr_robots_description/meshes/p3at_meshes/front_sonar.stl
  - [visual] package://amr_robots_description/meshes/p3at_meshes/top.stl
  - [visual] package://amr_robots_description/meshes/p3at_meshes/wheel.stl

### neato_botvac (diff, 2개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/neato_botvac`
  - [visual] package://neato_description/meshes/main_body.stl
  - [visual] package://neato_description/meshes/wheel.dae

### neobotix_mp500 (diff, 1개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/neobotix_mp500`
  - [visual] package://neo_mp_500-2/robot_model/meshes/MP-500-WHEEL.dae

### otto_1500 (diff, 6개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/otto_1500`
  - [visual] ../meshes/OTTO1500_bodylink.stl
  - [visual] ../meshes/OTTO1500_wheel_center.stl
  - [visual] ../meshes/OTTO1500_wheel_front_rear.stl
  - [visual] ../meshes/Platform_B.stl
  - [visual] ../meshes/lift.stl
  - [visual] ../meshes/lift_upper.stl

### otto_600 (diff, 4개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/otto_600`
  - [visual] ../Meshes/OTTO600_bodylink.stl
  - [visual] ../Meshes/OTTO600_platform_base.stl
  - [visual] ../Meshes/OTTO600_platform_upper.stl
  - [visual] ../Meshes/OTTO600_wheel.stl

### pal_ari (diff, 14개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/pal_ari`
  - [visual] package://ari_description/meshes/arm/arm_1.stl
  - [visual] package://ari_description/meshes/arm/arm_2.stl
  - [visual] package://ari_description/meshes/arm/arm_3.stl
  - [visual] package://ari_description/meshes/arm/arm_4.stl
  - [visual] package://ari_description/meshes/arm/arm_base.stl
  - [visual] package://ari_description/meshes/arm/hand_1.stl
  - [visual] package://ari_description/meshes/arm/hand_2.stl
  - [visual] package://ari_description/meshes/base/base.stl
  - [visual] package://ari_description/meshes/head/head_1.stl
  - [visual] package://ari_description/meshes/head/head_2.stl
  - [visual] package://ari_description/meshes/wheels/caster_left.stl
  - [visual] package://ari_description/meshes/wheels/caster_right.stl
  - [visual] package://ari_description/meshes/wheels/wheel_left.stl
  - [visual] package://ari_description/meshes/wheels/wheel_right.stl

### pal_tiago (diff, 16개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/pal_tiago`
  - [visual] package://pmb2_description/meshes/base/base.stl
  - [visual] package://pmb2_description/meshes/base/base_ring.stl
  - [visual] package://pmb2_description/meshes/objects/antenna.stl
  - [visual] package://pmb2_description/meshes/wheels/caster_1.stl
  - [visual] package://pmb2_description/meshes/wheels/caster_2.stl
  - [visual] package://pmb2_description/meshes/wheels/wheel.stl
  - [visual] package://tiago_description/meshes/arm/arm_1.stl
  - [visual] package://tiago_description/meshes/arm/arm_2.stl
  - [visual] package://tiago_description/meshes/arm/arm_3.stl
  - [visual] package://tiago_description/meshes/arm/arm_4.stl
  - [visual] package://tiago_description/meshes/arm/arm_5-wrist-2017.stl
  - [visual] package://tiago_description/meshes/arm/arm_6-wrist-2017.stl
  - [visual] package://tiago_description/meshes/head/head_1.stl
  - [visual] package://tiago_description/meshes/head/head_2.stl
  - [visual] package://tiago_description/meshes/torso/torso_fix.stl
  - [visual] package://tiago_description/meshes/torso/torso_lift_with_arm.stl

### robotis_turtlebot3 (diff, 4개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/robotis_turtlebot3`
  - [visual] package://turtlebot3_description/meshes/bases/waffle_pi_base.stl
  - [visual] package://turtlebot3_description/meshes/sensors/lds.stl
  - [visual] package://turtlebot3_description/meshes/wheels/left_tire.stl
  - [visual] package://turtlebot3_description/meshes/wheels/right_tire.stl

### robotnik_rbtheron (diff, 2개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/robotnik_rbtheron`
  - [visual] package://robotnik_description/meshes/bases/robotnik_logo_chasis.stl
  - [visual] package://robotnik_description/meshes/wheels/caster_wheel/caster_wheel.stl

### robotnik_rbwatcher (diff, 3개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/robotnik_rbwatcher`
  - [visual] package://robotnik_description/meshes/bases/rbwatcher/rbwatcher_chassis.dae
  - [visual] package://robotnik_description/meshes/bases/rbwatcher/rbwatcher_top_structure.dae
  - [visual] package://robotnik_description/meshes/wheels/rubber_wheel/rubber_wheel.dae

### robotnik_summit_xl (diff, 4개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/robotnik_summit_xl`
  - [visual] package://robotnik_sensors/meshes/hokuyo_ust_20lx.dae
  - [visual] package://summit_xl_description/meshes/bases/summit_xl_chassis.dae
  - [visual] package://summit_xl_description/meshes/wheels/rubber_wheel_left.dae
  - [visual] package://summit_xl_description/meshes/wheels/rubber_wheel_right.dae

### segway_rmp440le (diff, 2개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/segway_rmp440le`
  - [visual] package://rmp_description/meshes/rmp440le/base/base.stl
  - [visual] package://rmp_description/meshes/rmp440le/base/wheel.stl

### toyota_hsr (diff, 21개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/toyota_hsr`
  - [visual] package://hsrb_meshes/meshes/arm_v0/arm_flex_light.dae
  - [visual] package://hsrb_meshes/meshes/arm_v0/arm_roll_light.dae
  - [visual] package://hsrb_meshes/meshes/arm_v0/shoulder.dae
  - [visual] package://hsrb_meshes/meshes/base_v2/base_light.dae
  - [visual] package://hsrb_meshes/meshes/base_v2/body_light.dae
  - [visual] package://hsrb_meshes/meshes/base_v2/bumper.dae
  - [visual] package://hsrb_meshes/meshes/base_v2/torso_base.dae
  - [visual] package://hsrb_meshes/meshes/hand_v0/l_distal.dae
  - [visual] package://hsrb_meshes/meshes/hand_v0/l_proximal.dae
  - [visual] package://hsrb_meshes/meshes/hand_v0/palm_light.dae
  - [visual] package://hsrb_meshes/meshes/hand_v0/r_distal.dae
  - [visual] package://hsrb_meshes/meshes/hand_v0/r_proximal.dae
  - [visual] package://hsrb_meshes/meshes/head_v1/head_pan.dae
  - [visual] package://hsrb_meshes/meshes/head_v1/head_tilt.dae
  - [visual] package://hsrb_meshes/meshes/head_v1/head_upper.dae
  - [visual] package://hsrb_meshes/meshes/head_v1/tablet_base.dae
  - [visual] package://hsrb_meshes/meshes/sensors/laser.dae
  - [visual] package://hsrb_meshes/meshes/sensors/xtion.dae
  - [visual] package://hsrb_meshes/meshes/torso_v0/torso_light.dae
  - [visual] package://hsrb_meshes/meshes/wrist_v0/wrist_flex.dae
  - [visual] package://hsrb_meshes/meshes/wrist_v0/wrist_roll.dae

### yujin_kobuki (diff, 2개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/diff/yujin_kobuki`
  - [visual] package://kobuki_description/meshes/main_body.dae
  - [visual] package://kobuki_description/meshes/wheel.dae

### clearpath_ridgeback (omni, 7개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/clearpath_ridgeback`
  - [visual] package://clearpath_platform_description/meshes/r100/axle.stl
  - [visual] package://clearpath_platform_description/meshes/r100/body.stl
  - [visual] package://clearpath_platform_description/meshes/r100/end-cover.stl
  - [visual] package://clearpath_platform_description/meshes/r100/lights.stl
  - [visual] package://clearpath_platform_description/meshes/r100/rocker.stl
  - [visual] package://clearpath_platform_description/meshes/r100/side-cover.stl
  - [visual] package://clearpath_platform_description/meshes/r100/wheels/mecanum.stl

### galbot_s1 (omni, 41개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/galbot_s1`
  - [visual] meshes/visual/column_base_link.glb
  - [visual] meshes/visual/head_link1.glb
  - [visual] meshes/visual/head_link2.glb
  - [visual] meshes/visual/left_active_link1.glb
  - [visual] meshes/visual/left_active_link2.glb
  - [visual] meshes/visual/left_adapter_flange_link.glb
  - [visual] meshes/visual/left_arm_base_link.glb
  - [visual] meshes/visual/left_arm_camera_link.glb
  - [visual] meshes/visual/left_arm_link1.glb
  - [visual] meshes/visual/left_arm_link2.glb
  - [visual] meshes/visual/left_arm_link3.glb
  - [visual] meshes/visual/left_arm_link4.glb
  - [visual] meshes/visual/left_arm_link5.glb
  - [visual] meshes/visual/left_arm_link6.glb
  - [visual] meshes/visual/left_arm_link7.glb
  - [visual] meshes/visual/left_gripper_base_link.glb
  - [visual] meshes/visual/left_passive_link.glb
  - [visual] meshes/visual/omni_chassis_base_link.glb
  - [visual] meshes/visual/right_active_link1.glb
  - [visual] meshes/visual/right_active_link2.glb
  - [visual] meshes/visual/right_adapter_flange_link.glb
  - [visual] meshes/visual/right_arm_base_link.glb
  - [visual] meshes/visual/right_arm_camera_link.glb
  - [visual] meshes/visual/right_arm_link1.glb
  - [visual] meshes/visual/right_arm_link2.glb
  - [visual] meshes/visual/right_arm_link3.glb
  - [visual] meshes/visual/right_arm_link4.glb
  - [visual] meshes/visual/right_arm_link5.glb
  - [visual] meshes/visual/right_arm_link6.glb
  - [visual] meshes/visual/right_arm_link7.glb
  - [visual] meshes/visual/right_gripper_base_link.glb
  - [visual] meshes/visual/right_passive_link.glb
  - [visual] meshes/visual/torso_base_link.glb
  - [visual] meshes/visual/wheel_driving_link1.glb
  - [visual] meshes/visual/wheel_driving_link2.glb
  - [visual] meshes/visual/wheel_driving_link3.glb
  - [visual] meshes/visual/wheel_driving_link4.glb
  - [visual] meshes/visual/wheel_steering_link1.glb
  - [visual] meshes/visual/wheel_steering_link2.glb
  - [visual] meshes/visual/wheel_steering_link3.glb
  - [visual] meshes/visual/wheel_steering_link4.glb

### husarion_rosbot (omni, 5개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/husarion_rosbot`
  - [visual] package://rosbot_description/meshes/rosbot_xl/body.dae
  - [visual] package://rosbot_description/meshes/rosbot_xl/components/antenna.dae
  - [visual] package://rosbot_description/meshes/rosbot_xl/components/antenna_connector.dae
  - [visual] package://rosbot_description/meshes/rosbot_xl/wheel_a.dae
  - [visual] package://rosbot_description/meshes/rosbot_xl/wheel_b.dae

### kuka_kmriiwa (omni, 10개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/kuka_kmriiwa`
  - [visual] package://kmriiwa_description/meshes/iiwa14/visual/link_0.stl
  - [visual] package://kmriiwa_description/meshes/iiwa14/visual/link_1.stl
  - [visual] package://kmriiwa_description/meshes/iiwa14/visual/link_2.stl
  - [visual] package://kmriiwa_description/meshes/iiwa14/visual/link_3.stl
  - [visual] package://kmriiwa_description/meshes/iiwa14/visual/link_4.stl
  - [visual] package://kmriiwa_description/meshes/iiwa14/visual/link_5.stl
  - [visual] package://kmriiwa_description/meshes/iiwa14/visual/link_6.stl
  - [visual] package://kmriiwa_description/meshes/iiwa14/visual/link_7.stl
  - [visual] package://kmriiwa_description/meshes/kmp200/sensors/visual/sick_lms1xx.dae
  - [visual] package://kmriiwa_description/meshes/kmp200/wheels/omni_wheel_1.dae

### kuka_youbot (omni, 10개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/kuka_youbot`
  - [visual] package://youbot_description/meshes/sensors/hokuyo.dae
  - [visual] package://youbot_description/meshes/youbot_arm/arm0.dae
  - [visual] package://youbot_description/meshes/youbot_arm/arm1.dae
  - [visual] package://youbot_description/meshes/youbot_arm/arm2.dae
  - [visual] package://youbot_description/meshes/youbot_arm/arm3.dae
  - [visual] package://youbot_description/meshes/youbot_arm/arm4.dae
  - [visual] package://youbot_description/meshes/youbot_arm/arm5.dae
  - [visual] package://youbot_description/meshes/youbot_base/base.dae
  - [visual] package://youbot_description/meshes/youbot_gripper/finger.dae
  - [visual] package://youbot_description/meshes/youbot_gripper/palm.dae

### neobotix_mmo500 (omni, 1개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/neobotix_mmo500`
  - [visual] package://neo_mmo_500/robot_model/mmo_500/meshes/finger_palm_link.dae

### neobotix_mpo500 (omni, 1개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/neobotix_mpo500`
  - [visual] package://neo_mpo_500-2/robot_model/meshes/MPO-500-WHEEL.dae

### robotnik_rbkairos (omni, 4개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/robotnik_rbkairos`
  - [visual] package://robotnik_description/meshes/bases/rbkairos/rbkairos_top_cover.stl
  - [visual] package://robotnik_description/meshes/bases/robotnik_logo_chasis.stl
  - [visual] package://robotnik_description/meshes/wheels/mecanum_wheel/kairos_mecanum_wheel_1.dae
  - [visual] package://robotnik_description/meshes/wheels/mecanum_wheel/kairos_mecanum_wheel_2.dae

### robotnik_rbvogui (omni, 4개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/robotnik_rbvogui`
  - [visual] package://robotnik_description/meshes/bases/rbvogui/rbvogui_base_docking_contact.stl
  - [visual] package://robotnik_description/meshes/bases/rbvogui/rbvogui_base_logos.stl
  - [visual] package://robotnik_description/meshes/bases/rbvogui/rbvogui_leds.stl
  - [visual] package://robotnik_description/meshes/wheels/rubber_wheel/rubber_wheel.dae

### softbank_pepper (omni, 47개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/softbank_pepper`
  - [visual] package://pepper_meshes/meshes/1.0/HeadPitch.dae
  - [visual] package://pepper_meshes/meshes/1.0/HeadYaw.dae
  - [visual] package://pepper_meshes/meshes/1.0/HipPitch.dae
  - [visual] package://pepper_meshes/meshes/1.0/HipRoll.dae
  - [visual] package://pepper_meshes/meshes/1.0/KneePitch.dae
  - [visual] package://pepper_meshes/meshes/1.0/LElbowRoll.dae
  - [visual] package://pepper_meshes/meshes/1.0/LElbowYaw.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger11.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger12.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger13.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger21.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger22.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger23.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger31.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger32.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger33.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger41.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger42.dae
  - [visual] package://pepper_meshes/meshes/1.0/LFinger43.dae
  - [visual] package://pepper_meshes/meshes/1.0/LShoulderPitch.dae
  - [visual] package://pepper_meshes/meshes/1.0/LShoulderRoll.dae
  - [visual] package://pepper_meshes/meshes/1.0/LThumb1.dae
  - [visual] package://pepper_meshes/meshes/1.0/LThumb2.dae
  - [visual] package://pepper_meshes/meshes/1.0/LWristYaw.dae
  - [visual] package://pepper_meshes/meshes/1.0/RElbowRoll.dae
  - [visual] package://pepper_meshes/meshes/1.0/RElbowYaw.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger11.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger12.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger13.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger21.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger22.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger23.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger31.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger32.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger33.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger41.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger42.dae
  - [visual] package://pepper_meshes/meshes/1.0/RFinger43.dae
  - [visual] package://pepper_meshes/meshes/1.0/RShoulderPitch.dae
  - [visual] package://pepper_meshes/meshes/1.0/RShoulderRoll.dae
  - [visual] package://pepper_meshes/meshes/1.0/RThumb1.dae
  - [visual] package://pepper_meshes/meshes/1.0/RThumb2.dae
  - [visual] package://pepper_meshes/meshes/1.0/RWristYaw.dae
  - [visual] package://pepper_meshes/meshes/1.0/Torso.dae
  - [visual] package://pepper_meshes/meshes/1.0/WheelB.dae
  - [visual] package://pepper_meshes/meshes/1.0/WheelFL.dae
  - [visual] package://pepper_meshes/meshes/1.0/WheelFR.dae

### willowgarage_pr2 (omni, 19개 누락)
- asset_dir: `/workspace/eatrav/data/sim/assets/robot/wheeled/omni/willowgarage_pr2`
  - [visual] package://pr2_description/meshes/base_v0/base.dae
  - [visual] package://pr2_description/meshes/base_v0/caster.stl
  - [visual] package://pr2_description/meshes/base_v0/wheel.dae
  - [visual] package://pr2_description/meshes/forearm_v0/forearm.dae
  - [visual] package://pr2_description/meshes/forearm_v0/wrist_flex.dae
  - [visual] package://pr2_description/meshes/forearm_v0/wrist_roll.stl
  - [visual] package://pr2_description/meshes/gripper_v0/gripper_palm.dae
  - [visual] package://pr2_description/meshes/gripper_v0/l_finger.dae
  - [visual] package://pr2_description/meshes/gripper_v0/l_finger_tip.dae
  - [visual] package://pr2_description/meshes/head_v0/head_pan.dae
  - [visual] package://pr2_description/meshes/head_v0/head_tilt.dae
  - [visual] package://pr2_description/meshes/shoulder_v0/shoulder_lift.dae
  - [visual] package://pr2_description/meshes/shoulder_v0/shoulder_pan.dae
  - [visual] package://pr2_description/meshes/shoulder_v0/upper_arm_roll.stl
  - [visual] package://pr2_description/meshes/tilting_laser_v0/tilting_hokuyo.dae
  - [visual] package://pr2_description/meshes/torso_v0/torso_lift.dae
  - [visual] package://pr2_description/meshes/upper_arm_v0/elbow_flex.dae
  - [visual] package://pr2_description/meshes/upper_arm_v0/forearm_roll.stl
  - [visual] package://pr2_description/meshes/upper_arm_v0/upper_arm.dae

---

## 4. 로봇 후보 목록 (회사 - 로봇명, 판단 없이 전수 나열)

### 4.1 Wheeled

- Clearpath Robotics: Husky, Husky A300 AMP, Jackal, Warthog, Ridgeback, Boxer, Dingo
- Robotnik: Summit XL, RB-1, RB-Kairos, RB-Theron, RB-Vogui, RB-Vogui XL, RB-Robout, RB-Watcher, RB-Summit, RB-Summit-Steel
- AgileX Robotics: Scout 2.0, Scout Mini, Hunter, Hunter SE, Ranger, Ranger Mini 3.0, Bunker, Bunker Mini 2.0, Bunker Pro, Tracer 2.0, Limo
- MiR (Mobile Industrial Robots): MiR100, MiR200, MiR250, MiR500, MiR600, MiR1000, MiR1350, MiR1200 Pallet Jack, MC600
- ROBOTIS: TurtleBot3, TurtleBot4
- PAL Robotics: TIAGo, TIAGo++, TIAGo OMNI++, TIAGo Base, TIAGo Pro, ARI, StockBot
- Hello Robot: Stretch RE1, Stretch(2023), Stretch 3, Stretch 4
- iRobot: Roomba, Create, Create3, PackBot, Braava, Braava jet
- Toyota: HSR
- Husarion: Panther, Lynx, ROSbot 2, ROSbot 2 Pro, ROSbot 2R, ROSbot 3, ROSbot 3 Pro, ROSbot XL
- Fetch Robotics: Fetch, Freight100, Freight500, Freight1500, RollerTop, CartConnect, HMIShelf
- MobileRobots/Adept (Omron Adept MobileRobots): Pioneer 1, Pioneer 2, Pioneer 3-DX, Pioneer 3-AT, Pioneer LX, PowerBot, PatrolBot, PeopleBot, AmigoBot, Seekur, Seekur Jr
- SoftBank Robotics: Pepper, Whiz
- Omron: LD-60, LD-90, LD-250, HD-1500
- Neobotix: MP-400, MP-500, MPO-500, MPO-700, MMO-700
- KUKA: KMR QUANTEC, KMR iiwa, KMR iisy, KMR iisy CR, KMP 1500P, KMP 600-S diffDrive, youBot
- Segway Robotics: Loomo, Nova Carter
- Locus Robotics: LocusBot, Locus Origin, Locus Vector, Locus Array, Locus Max
- 6 River Systems: Chuck, Chuck+
- Seegrid: Palion Lift, Palion Pallet Truck, Palion Tow Tractor, Lift RS1
- inVia Robotics: Picker
- Waypoint Robotics: Vector, MAV3K
- Festo: Robotino, Robotino R, Robotino XT, Robotino XXT
- Boston Dynamics: Handle, Stretch
- Amazon Robotics: Kiva, Hercules, Pegasus, Xanthus, Proteus, Titan
- Geek+: P500, P1200, MP1000R, Bin Mover M600, Bin Mover M100
- Hai Robotics: HaiPick A42, HaiPick A42T, HaiPick A42-E6S, HaiPick A3, A42D
- Vecna Robotics: Vecna AFL, Vecna ATG, Vecna APT, Vecna CPJ, BEAR, QC Bot
- OTTO Motors: OTTO 100, OTTO 600, OTTO 1200, OTTO 1500, OTTO Lifter
- idealworks: iw.hub
- NVIDIA: Nova Carter, JetBot
- Savioke: Relay
- Aethon: TUG, TUG T3
- Simbe Robotics: Tally, Tally 3.0
- Diligent Robotics: Moxi, Moxi 2.0
- Bossa Nova Robotics: (매장 재고스캔 로봇)
- Piaggio Fast Forward: gita, gitamini, kilo
- Yujin Robot: Kobuki, GoCart120, GoCart180, GoCart200 Omni, GoCart250
- Cobalt Robotics: Cobalt
- Starship Technologies: Starship robot
- Nuro: R2
- Kiwibot: Kiwibot
- Cartken: Model C, Cartken Hauler
- Pudu Robotics: BellaBot, KettyBot, HolaBot, CC1
- Keenon Robotics: DINERBOT T3/T5/T6/T8/T9/T9 Pro/T10/T11, BUTLERBOT W3, KLEENBOT C20/C25/C30/C40, M1, M2, G1, G2, S100
- Bear Robotics: Servi, Servi mini, Servi Plus, Servi Q, Carti
- LG: CLOi ServeBot, CLOi GuideBot, CLOi CarryBot
- GreyOrange: Butler, Butler XL, Ranger GTP, Butler PickPal
- Exotec: Skypod
- Dexory: Grande 44
- Rapyuta Robotics: PA-AMR, PA-AMR High Capacity Model
- Gideon Brothers: TREY
- MetraLabs: SCITOS G5, SCITOS A5, SCITOS X3, TORY, MORPHIA, CARY, STERYBOT
- AGILOX: AGILOX ONE, OFL, OCF, ODM
- Quicktron: Box Picker, M-series
- Syrius Robotics: FlexSwift MAX, FlexPorter
- ASTI Mobile Robotics / Kivnon: K03 Twister, K50 Pallet Truck, K55 Pallet Stacker
- Cyngn: DriveMod Tugger, DriveMod Forklift
- ForwardX Robotics: Max 1500-L Slim, Flex 300-L Mini, Flex 60-SW, Apex series
- IAM Robotics: Swift, Bolt
- Swisslog: IntraMove AMR 600, IntraMove AMR 1500, IntraMove AMR 3000
- Balyo: VEENY, REACHY, LOWY CB, TUGGY, STACKY
- Avidbots: Neo, Neo 2W, Kas
- Gausium: Phantas, Scrubber 50, Scrubber 75, Omnie, Vacuum 40, Beetle
- Double Robotics: Double, Double 2, Double 3
- Ava Robotics: Ava, Ava 500
- Ecovacs: DEEBOT N20, T30S, T50 Max Pro, X8, X8 Pro, X9, X11
- Neato Robotics: D10
- Kärcher: KIRA B 50, KIRA B 200, KIRA BR 200, KIRA BD 200, KIRA CV 50
- Milvus Robotics: SEIT300, SEIT500, MRP2, Robin
- Panasonic: HOSPI, HOSPI-Rimo
- Woowa Brothers: Dilly Drive, Dilly Tower, Dilly X3-R
- Naio Technologies: Oz, Dino, Orio, Ted, Jo
- Willow Garage: PR2
- Dusty Robotics: FieldPrinter, FieldPrinter 2
- Honda: UNI-CUB, UNI-CUB β
- YOUIBOT: TRANS
- Techmetics: TRV Mini, TRV Lifter, TRV Mega
- Magazino: TORU, SOTO
- ABB: Flexley Tug, Flexley Mover
- Rainbow Robotics: RB-Y1
- Sphero: RVR, BOLT, Mini, SPRK+

### 4.2 Legged

- Unitree: Laikago, AlienGo, A1, Go1, Go2, B1, B2, B2-W, As2-W, BenBen
- ANYbotics: StarlETH, ANYmal, ANYmal B, ANYmal C, ANYmal D, ANYmal X
- Boston Dynamics: BigDog, LittleDog, Cheetah, WildCat, LS3/AlphaDog, RiSE, Spot, SpotMini
- Deep Robotics: Jueying X20, X30, Lite2, Lite3, M20, Lynx M20, DR01, DR02
- Ghost Robotics: Minitaur, Vision 60, Spirit 40
- LimX Dynamics: W1, P1, CL-1
- ODRI: Solo8, Solo12, Bolt
- IIT: HyQ, MiniHyQ, HyQ2Max, HyQ2Centaur, HyQReal, Centauro
- MAB Robotics: Honey Badger
- Xiaomi: CyberDog, CyberDog 2
- Petoi: Bittle, Bittle X, Nybble, Nybble Q
- MangDang: Mini Pupper, Mini Pupper 2
- Trossen Robotics/Interbotix: PhantomX Hexapod MK-I, MK-II, MK-III
- Weilan: AlphaDog, BabyAlpha
- Rainbow Robotics: RB-Q, RBQ-10
- Festo: BionicWheelBot, BionicKangaroo, BionicANTs
- Sony: AIBO(ERS-110/111/210/220/7/7M3/1000)
- Hyundai: Elevate, TIGER
- KAIST: Raibo, Raibo2, HOUND
- Google DeepMind: Barkour
- MIT: Cheetah, Cheetah 2, Cheetah 3, Mini Cheetah, HERMES
- Rodney Brooks Lab(MIT): Genghis
- Stanford: Stanford Doggo, Pupper
- Swiss-Mile: Swiss-Mile robot
- Tencent Robotics X: Max, Ollie, The Five(Xiao Wu)
- XPeng Robotics: Xiaobailong, Unicorn
- AgiBot/AgiQuad: AgiQuad 시리즈
- DaxAI Robotics: Qiji X1
- Vincross: HEXA
- Hiwonder: PuppyPi, ROSPug, JetHexa
- Keybotic: Keyper
- Hyperever: Proteo
- MicroRoboTech: Move New T1
- Aeroarc: MULES
- Bhairav Robotics: Shvana
- Diden Robotics: MARVEL
- DFKI: SpaceClimber, Charlie, MANTIS, CREX
- FZI: LAURON I, II, III, IV
- 도쿄공업대(Hirose Lab): TITAN-VIII, TITAN-IX, TITAN-XI, TITAN-XII, TITAN-XIII, TITAN-E1, ASTERISK
- UPenn(Kod*Lab/GRASP): RHex, X-RHex, Minitaur, Canid
- NASA JPL: ATHLETE, RoboSimian, LEMUR, LEMUR 3
- Florida A&M University: SCARAB
- Innovation First(Hexbug): Hexbug 시리즈
- Case Western Reserve University: Whegs I, Whegs II, Autonomous Whegs II
- Ohio State University: Adaptive Suspension Vehicle(ASV)
- Indiana University: Stiquito
- WowWee: Roboquad, CHiP, Robopet, Wrex the Dawg
- Silverlit Electronics: I-Cybie
- Sega/Tiger Electronics: iDog, Poo-Chi
- Fisher-Price: Rocket the Wonder Dog
- Tomy: Spotbot, Omnibot
- Ideal Toy Company: Gaylord
- Dasatech: Genibo
- Zizzle: Lucky the Incredible Wonder Pup
- Trendmasters: Big Scratch & Lil' Scratch
- Tombot: Jennie
- Joinmax: JM-DOG-001

### 4.3 Humanoid

(다리)=이족보행, (바퀴)=바퀴형 베이스

- PAL Robotics: REEM-A(다리), REEM-B(다리), REEM-C(다리), REEM(바퀴), TALOS(다리), Kangaroo(다리), ARI(바퀴)
- Unitree: H1(다리), H1-2(다리), H2(다리), H2 Plus(다리), G1(다리), R1(다리)
- ROBOTIS: DARwIn-OP(다리), OP2(다리), OP3(다리), THORMANG(다리), THORMANG3(다리), AI Worker/FFW-SG2(바퀴)
- SoftBank/Aldebaran: NAO(다리), Pepper(바퀴), Romeo(다리)
- Fourier Intelligence: GR-1(다리), GR-2(다리), GR-3(다리)
- Booster Robotics: T1(다리), T2(다리), K1(다리)
- IIT: iCub(다리), COMAN(다리), WALK-MAN(다리), ergoCub(다리)
- Agility Robotics: Cassie(다리), Digit(다리)
- Boston Dynamics: Atlas 유압식(다리), Atlas 전동식(다리)
- NASA: Valkyrie/R5(다리), Robonaut 2(다리)
- Apptronik: Apollo(다리), Apollo 2(다리)
- Figure AI: Figure 01(다리), Figure 02(다리), Figure 03(다리)
- 1X Technologies: EVE(바퀴), NEO(다리), NEO Beta(다리), NEO Gamma(다리)
- Tesla: Optimus Gen1(다리), Gen2(다리), Gen3(다리)
- UBTech: Walker(다리), Walker X(다리), Walker C(다리), Walker C1(다리), Walker S(다리), Walker S1(다리), Walker S2(다리), Cruzr S2(바퀴), Cruzr Y1(바퀴)
- Xiaomi: CyberOne(다리), CyberOne v2(다리)
- Engineered Arts: Ameca(고정형/바퀴), Ameca Desktop(고정형)
- Sanctuary AI: Phoenix(다리), Phoenix 3.2(다리)
- Agibot/Zhiyuan: Yuanzheng A2(다리), A2-W(다리), A2-Max(다리), A3(다리), Lingxi X1(다리), Lingxi X1-W(다리), Lingxi X2(다리), G2 Genie(다리), G2 Max(다리), X1(다리), X2(다리), RAISE A1(다리)
- Galbot: G1(바퀴), ET1(다리), S1(다리)
- RobotEra: STAR1(다리), L7(다리), M7(다리)
- Westwood Robotics: BRUCE(다리), THEMIS V2(다리)
- CAST/University of Tehran: Surena I/II/III/IV/V(다리)
- KAIST/HUBO Lab: KHR-1(다리), KHR-2(다리), KHR-3/HUBO(다리), HUBO2(다리), Albert HUBO(다리), DRC-HUBO(다리/바퀴 겸용)
- Honda: E0~E6(다리), P1~P3(다리), ASIMO(다리)
- Kawasaki: Kaleido(다리)
- Toyota: T-HR3(다리)
- Sony: QRIO(다리)
- LG: CLOiD(바퀴)
- Xpeng: PX5(다리), Iron(다리)
- Rainbow Robotics: HUBO-2(다리), DRC-HUBO(다리), FX-2(다리), RB-Y1(바퀴)
- Deep Robotics: DR01(다리), DR02(다리)
- Kepler: Forerunner K1(다리), K2(다리), S1(다리), D1(다리)
- EngineAI: SE01(다리), PM01(다리), T800(다리)
- LimX Dynamics: Oli(다리), CL-1(다리), CL-2(다리), CL-3(다리), TRON 1(다리/바퀴), TRON2A(다리/바퀴)
- MagicLab: MagicBot(다리), MagicBot X1(다리), MagicBot Z1(다리)
- PNDbotics: Adam(다리), Adam SP(다리), Adam Lite(다리), Adam-U(고정형)
- Hanson Robotics: Sophia(다리), Han(고정형)
- Clone Robotics: Clone Alpha(다리)
- Neura Robotics: 4NE1(다리), 4NE1 Mini(다리)
- Astribot/Stardust Intelligence: S1(바퀴), T1(다리)
- UC Berkeley: Berkeley Humanoid(다리), Berkeley Humanoid Lite(다리)
- X-Humanoid(베이징휴머노이드혁신센터): Tiangong 1.0 Lite(다리), Tiangong/Tien Kung 2.0(다리), Tien Kung 3.0(다리), Tiangong Omni(다리)
- OpenLoong(상하이휴머노이드혁신센터): Qinglong V3.0(다리)
- NAVIAI(저장휴머노이드혁신센터): Navigator 1(다리), Navigator 2(다리)
- Beyond Imagination: Beomni(바퀴)
- Leju Robotics: Kuafu/Kuavo(다리), Kuavo-5(다리), Kuavo-my(다리), AELOS(다리), Roban2(다리)
- Mentee Robotics: MenteeBot(다리), MenteeBot V3.0(다리)
- Pollen Robotics: Reachy 2(바퀴)
- Dexmate: Vega(바퀴)
- Kinisi Robotics: KR1/Kinisi 01(바퀴)
- Mirsee Robotics: MH3(바퀴)
- Borg Robotics: Borg 01(다리/바퀴 전환형)
- Dobot/Shenzhen Yuejiang: Atom(다리), Atom Max(다리)
- Foundation Future Industries: Phantom MK-1(다리)
- Humanoid/SKL Robotics: HMND 01(다리), HMND 01 Alpha(다리)
- Techman Robot: TM Xplore 1(다리)
- Tencent Robotics X: Xiao Liu(다리)
- HONOR: Lightning(다리)
- Oversonic Robotics: RoBee(다리)
- Galaxea Dynamics: R1 PRO(바퀴)
- Tokyo Robotics: Torobo(바퀴)
- TeknTrash: ALPHA(바퀴)
- PaXini: TORA-ONE(바퀴), TORA DoubleOne(바퀴)
- DLR(독일항공우주센터): Rollin' Justin(바퀴), TORO(다리)
- AIST/Kawada: HRP-1, HRP-2, HRP-2P, HRP-3, HRP-3P, HRP-4, HRP-4C(모두 다리)
- Waseda University: WABOT-1(다리), WABIAN 시리즈(다리), Kobian(다리), Twendy-One(바퀴)
- Fujitsu: HOAP-1(다리), HOAP-2(다리), HOAP-3(다리)
- University of Bonn: NimbRo-OP(다리)
- INRIA: Poppy(다리)
- Enchanted Tools: Mirokaï(바퀴)
- Sulu.be: Steve(고정형)
- Mitsubishi Heavy Industries: Wakamaru(바퀴)
- Hitachi: WHL-11(다리)
- Robosen: Interstellar Scout K1(다리), K1 Pro(다리)
- Faraday Future: FF Master(바퀴 추정)
- Generative Bionics: GENE.01(다리 추정)
- iHub Robotics: Tara Gen1(바퀴), Tara Greet/Lean/Care(바퀴)
- Muks Robotics: Spaceo Pro(다리 추정), Spaceo M1(다리 추정), Spaceo Prime(다리 추정)
- Matrix Robotics: MATRIX-1(다리), MATRIX-3(다리)
- WIRobotics: ALLEX(바퀴 추정)
- 그 외 명칭만 확인(다리/바퀴 미검증): AEI Robot(Alice, Alice M1), AKINROBOTICS(AKINCI-5), Andromeda Robotics(Abi), AtaroBot(AtaroBot), CASBOT(CASBOT 02, W1), CasiVision(CASIVIBOT), Changingtek(X2), Cyan Robotics(Orca), Digit Robotics-중국(Nezha P01, Xialan S0, XiaQi), Donut Robotics(Cinnamon 1), EIR Technology(SkyWalker 2), Elu.AI(AstroD AD-01), Fauna Robotics(Sprout), Futuring Robot(Futuring 2), GigaAI(Maker H01), Haier(HIVA Haiwa), Hexagon(AEON), Holiday Robotics(Friday), KEENON Robotics(XMAN-R1), Lanxin Robotics(VB1-I, VB2, VersaBot VB-1), LINKERBOT(L30/O6/R30), Lumos Robotics(LUS2, NIX), Mand.ro(Mark 7), MenloAI(Asimov 1, Asimov 2), Midea(MIRO), Moon Dynamics(L1), Noble Machines(Moby), Noetix Robotics(N2, Dora, Hobbs), Nori Robotics(NORI L3), O-ID(Modular), OceanTrix Robotics(R1, R2), ORBIT Robotics(HELIOS), PrimeBOT(Q1, T1), PsiBot(E1, H1, V1), Pudu Robotics(D9, D7), Reflex Robotics(Reflex), Robbyant(R1), Robo Robotics(ROBO-T1), RoboForce(TITAN), Roboligent(Robin), Ruiyan(RY-H1), Simplexity Robotics(I-Series), Spirit AI(Moz1), Sunday Robotics(MEMO), TARS Robotics(TARS), TOPSTAR Group(Xiao Tuo), TwoLabs(Tobi), Ultra Robotics(TITAN/OP1), UMA(Northstar), Unix Group(Martian, Panther, Wanda 2.0), VinMotion(Motion 2), Vittorio Lumare(YEAH), Walden Robotics(Walden), WorkFar Technologies(Syntro), X Square Robot(ArtiXon), Xynova(Flex 2), XYZ(DEUX), Zerith Robotics(H1, Z1), Zeroth Robotics(Sean), WL Robotics(Totan O1)

---

## 5. 로봇별 오픈소스 RL 보행 학습 리포지토리

- 조사일: 2026-09-06
- 목적: 워크스페이스 `data/sim/usd/real_robot/legged/` 34종(multi-legged 16 + humanoid 18)별로, 보행 정책 학습 코드가 공개된 오픈소스 리포지토리 링크를 전수 수집
- 범위: **링크 수집만** 함. 리포 내부 구조·학습 방식·하이퍼파라미터·라이선스 적합성은 분석하지 않음(별도 작업)
- 판정 기준: 해당 로봇이 학습 대상으로 명시된(태스크 이름·설정 파일·README에 등장) 리포만 "로봇별"에 적음. 로봇 특정 없이 프레임워크만 제공하는 것은 «5.1 공통 프레임워크»로 분리
- 검증 수준: 검색 + 주요 리포는 README 직접 확인. USD/URDF/MJCF 자산이 이 리포들 안에 함께 들어 있는 경우가 많아 «1~2장»의 자산 출처와 겹칠 수 있음

### 5.1 여러 로봇을 함께 다루는 공통 프레임워크

| 리포 | 링크 | 기반 | 워크스페이스 로봇 중 커버(검색·README 기준) |
|---|---|---|---|
| leggedrobotics/legged_gym | https://github.com/leggedrobotics/legged_gym | Isaac Gym | anymal_b, anymal_c, a1 (기본 포함) |
| leggedrobotics/rsl_rl | https://github.com/leggedrobotics/rsl_rl | PPO 알고리즘 라이브러리 | (위·아래 대부분의 학습기) |
| isaac-sim/IsaacLab | https://github.com/isaac-sim/IsaacLab | Isaac Sim | anymal_b/c/d, a1, go1, go2, spot, h1, g1, digit (velocity locomotion 기본 포함) |
| fan-ziqi/robot_lab | https://github.com/fan-ziqi/robot_lab | Isaac Lab | anymal_d, go2, go2w, b2, a1, lite3, m20, booster_t1, robotera_star1(XBot) |
| fan-ziqi/rl_sar | https://github.com/fan-ziqi/rl_sar | 배포(sim/real) | lite3 등 4족·휴머노이드 |
| unitreerobotics/unitree_rl_gym | https://github.com/unitreerobotics/unitree_rl_gym | Isaac Gym | go2, g1, h1, h1_2 |
| unitreerobotics/unitree_rl_lab | https://github.com/unitreerobotics/unitree_rl_lab | Isaac Lab | go2, g1(29dof), h1 |
| unitreerobotics/unitree_rl_mjlab | https://github.com/unitreerobotics/unitree_rl_mjlab | MuJoCo | go2, g1, h1_2, h2 |
| DeepRoboticsLab/rl_training | https://github.com/DeepRoboticsLab/rl_training | Isaac Lab (2.3.x) | lite3, m20, dr02 — BSD-3-Clause/Apache-2.0 |
| limxdynamics/tron2-rl-isaaclab | https://github.com/limxdynamics/tron2-rl-isaaclab | Isaac Lab | tron2a_sf, tron2a_wf |
| InternRobotics/HIMLoco | https://github.com/InternRobotics/HIMLoco | Isaac Gym (legged_gym) | a1, aliengo, go1 |
| HybridRobotics/GenLoco | https://github.com/HybridRobotics/GenLoco | Isaac Gym | a1 등 4족 morphology 일반화 정책 |
| LeCAR-Lab/HumanoidVerse | https://github.com/LeCAR-Lab/HumanoidVerse | IsaacGym/IsaacSim/Genesis | g1, h1 |
| roboterax/humanoid-gym | https://github.com/roboterax/humanoid-gym | Isaac Gym | robotera_star1(XBot-S/L) |
| BoosterRobotics/booster_gym | https://github.com/BoosterRobotics/booster_gym | Isaac Gym | booster_t1 |
| BoosterRobotics/booster_train | https://github.com/BoosterRobotics/booster_train | Isaac Lab | booster_t1 |

### 5.2 multi-legged (16종)

- **anybotics_anymal_b** — https://github.com/leggedrobotics/legged_gym (기본 포함) ; https://github.com/isaac-sim/IsaacLab (기본 포함)
- **anybotics_anymal_c** — https://github.com/leggedrobotics/legged_gym (기본 포함) ; https://github.com/isaac-sim/IsaacLab (기본 포함) ; https://github.com/ETH-PBL/elmap-rl-controller
- **anybotics_anymal_d** — https://github.com/isaac-sim/IsaacLab (기본 포함) ; https://github.com/fan-ziqi/robot_lab
- **bostondynamics_spot** — https://github.com/isaac-sim/IsaacLab (spot flat/rough 기본 포함) ; https://github.com/OpenQuadruped/spot_mini_mini ; https://github.com/gilbertgonz/spot_reinforcement_learning ; RAI Institute 엔드투엔드 파이프라인 — 논문 https://arxiv.org/abs/2504.17857 , BD 블로그 https://bostondynamics.com/blog/starting-on-the-right-foot-with-reinforcement-learning/
- **deeprobotics_lite3** — https://github.com/DeepRoboticsLab/rl_training (공식, 태스크 `Rough-Deeprobotics-Lite3-v0`) ; https://github.com/DeepRoboticsLab/Lite3_rl_deploy (배포) ; https://github.com/DeepRoboticsLab/Lite3_MotionSDK ; https://github.com/fan-ziqi/robot_lab ; https://github.com/fan-ziqi/rl_sar
- **deeprobotics_m20** — https://github.com/DeepRoboticsLab/rl_training (공식, 태스크 `Rough-Deeprobotics-M20-v0`) ; https://github.com/fan-ziqi/robot_lab ; https://github.com/lbnmahs/quadrrl
- **deeprobotics_m20_piper** — https://github.com/DeepRoboticsLab/rl_training (M20 베이스, piper=로봇팔 결합 변형 — 전용 태스크는 미확인)
- **deeprobotics_m20s** — https://github.com/DeepRoboticsLab/rl_training (M20 계열 — 전용 태스크는 미확인)
- **deeprobotics_x30** — 전용 RL 학습 리포 **미확인** (DeepRobotics 공식 `rl_training`은 Lite3/M20/DR02만 지원, X30은 순찰·검사용). 참고 org: https://github.com/DeepRoboticsLab
- **unitree_a1** — https://github.com/leggedrobotics/legged_gym (기본 포함) ; https://github.com/isaac-sim/IsaacLab (기본 포함) ; https://github.com/unitreerobotics/unitree_rl_gym ; https://github.com/InternRobotics/HIMLoco ; https://github.com/HybridRobotics/GenLoco ; https://github.com/silvery107/rl-mpc-locomotion
- **unitree_aliengo** — https://github.com/InternRobotics/HIMLoco (학습 + 실기 배포) ; https://github.com/silvery107/rl-mpc-locomotion
- **unitree_b2** — https://github.com/fan-ziqi/robot_lab (태스크 `RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0`)
- **unitree_go1** — https://github.com/isaac-sim/IsaacLab (기본 포함) ; https://github.com/unitreerobotics/unitree_rl_gym ; https://github.com/Improbable-AI/walk-these-ways ; https://github.com/InternRobotics/HIMLoco ; https://github.com/Teddy-Liao/walk-these-ways-go2
- **unitree_go2** — https://github.com/isaac-sim/IsaacLab (기본 포함) ; https://github.com/unitreerobotics/unitree_rl_gym ; https://github.com/unitreerobotics/unitree_rl_lab ; https://github.com/unitreerobotics/unitree_rl_mjlab ; https://github.com/fan-ziqi/robot_lab ; https://github.com/wty-yy/go2_rl_gym ; https://github.com/abizovnuralem/go2_omniverse
- **unitree_go2w** — https://github.com/fan-ziqi/robot_lab (휠-레그 변형)
- **unitree_laikago** — 전용 RL 학습 리포 **미확인**. 모델/자산: https://github.com/unitreerobotics/unitree_ros ; morphology 일반화 정책에 포함: https://github.com/HybridRobotics/GenLoco

### 5.3 humanoid (18종)

- **agibot_a2d** — AgiBot A2/A2D 전용 리포는 AgibotTech org에 **없음**. 가장 가까운 것: https://github.com/AgibotTech/agibot_x1_train (AgiBot X1용 이족보행 RL 학습) ; 모델: https://github.com/AgibotTech/agibot_x2_urdf
- **agility_digit** — https://github.com/isaac-sim/IsaacLab (Digit rough terrain 기본 포함, 폐루프 킨매틱 체인) ; https://github.com/rohanpsingh/LearningHumanoidWalking
- **boosterrobotics_t1** — https://github.com/BoosterRobotics/booster_gym (공식, Isaac Gym) ; https://github.com/BoosterRobotics/booster_train (공식, Isaac Lab) ; https://github.com/NaoHTWK/htwk-gym ; https://github.com/fan-ziqi/robot_lab
- **deeprobotics_dr02** — https://github.com/DeepRoboticsLab/rl_training (공식, 태스크 `Amp-Flat-Deeprobotics-DR02-v0` — AMP 기반)
- **fourier_gr1** — https://github.com/zixuan417/smooth-humanoid-locomotion (GR1T2 학습+배포, legged_gym+rsl_rl) ; https://homietele.github.io/ (HOMIE, Isaac Gym) ; https://github.com/junfeng-long/PIM (Perceptive Internal Model)
- **ihmcrobotics_valkyrie** — RL 학습 리포 **미확인** (공개 스택은 모멘텀 기반 모델 제어). 참고: https://github.com/ihmcrobotics/ihmc-open-robotics-software , https://github.com/ihmcrobotics/valkyrie
- **limx_hu_d03** — https://github.com/limxdynamics/humanoid-rl-deploy-ros (RL 배포 프레임워크) ; 모델: https://github.com/limxdynamics/humanoid-description . 전용 학습(Isaac) 리포는 미확인
- **limx_hu_d04** — 위 `limx_hu_d03`와 동일 (LimX humanoid 공용 배포·모델 리포)
- **limx_tron2a_sf** — https://github.com/limxdynamics/tron2-rl-isaaclab (공식, SF/WF 변형 지원) ; https://github.com/limxdynamics/tron2-mujoco-sim ; 이전 세대: https://github.com/limxdynamics/tron1-rl-isaacgym (= pointfoot-legged-gym)
- **limx_tron2a_wf** — 위 `limx_tron2a_sf`와 동일 리포 (`tron2-rl-isaaclab`가 WF=바퀴발 변형 포함)
- **onex_neo** — RL 학습 리포 **미확인** (1X Technologies "Redwood" 컨트롤러 비공개)
- **robotera_star1** — https://github.com/roboterax/humanoid-gym ; https://github.com/biomechatronics001/humanoid-gym-RobotEra ; https://github.com/fan-ziqi/robot_lab (RobotEra XBot) ; https://github.com/roboterax (ros2_sdk에 star1 예제)
- **sanctuaryai_phoenix** — RL 학습 리포 **미확인** (Sanctuary AI "Carbon" 비공개)
- **unitree_g1** — https://github.com/unitreerobotics/unitree_rl_gym ; https://github.com/unitreerobotics/unitree_rl_lab ; https://github.com/unitreerobotics/unitree_rl_mjlab ; https://github.com/isaac-sim/IsaacLab (기본 포함) ; https://github.com/LeCAR-Lab/HumanoidVerse ; https://github.com/zitongbai/legged_lab ; https://github.com/HusseinLezzaik/unitree-g1-bipedal-rl-walk
- **unitree_h1** — https://github.com/unitreerobotics/unitree_rl_gym ; https://github.com/unitreerobotics/unitree_rl_lab ; https://github.com/isaac-sim/IsaacLab (기본 포함) ; https://github.com/LeCAR-Lab/HumanoidVerse ; https://github.com/dancher00/him-humanoid-h1
- **unitree_h1_2** — https://github.com/unitreerobotics/unitree_rl_gym ; https://github.com/unitreerobotics/unitree_rl_mjlab
- **unitree_h2** — https://github.com/unitreerobotics/unitree_rl_mjlab (공식, MuJoCo — H2 지원 명시)
- **xiaopeng_px5** — RL 학습 리포 **미확인** (XPeng Robotics 비공개)

### 5.4 오픈소스 RL 학습 코드 미확인 (제조사 비공개 스택 / 전용 리포 없음)

- deeprobotics_x30 — 제조사 공개 RL은 Lite3/M20/DR02만
- ihmcrobotics_valkyrie — 공개 스택이 모델 기반 제어
- limx_hu_d03 / limx_hu_d04 — RL 배포 프레임워크만 공개, 학습 리포 미확인
- unitree_laikago — 전용 학습 리포 없음(자산·morphology 일반화 리포만)
- onex_neo (1X NEO) / sanctuaryai_phoenix / xiaopeng_px5 — 제조사 비공개
- agibot_a2d — A2/A2D 전용 없음, X1 학습 리포로 대체

### 5.5 참고 aggregator

- https://github.com/gaiyi7788/awesome-legged-locomotion-learning
- https://github.com/apexrl/awesome-rl-for-legged-locomotion
- https://github.com/clearlab-sustech/Awesome-Legged-Robot-Learning
- https://github.com/shaoxiang/awesome-unitree-robots
- https://github.com/XinLang2019/awesome-wheeled-legged (go2w, m20_piper 관련)
- https://github.com/robotlearning123/awesome-isaac-gym
