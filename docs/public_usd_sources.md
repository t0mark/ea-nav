# 공개 USD 소스 조사

- 조사일: 2026-09-04 ~ 2026-09-05
- 목적: URDF -> USD 변환 없이, 제조사·플랫폼이 직접 공개한 USD를 로봇 타입(바퀴형/다족보행/휴머노이드)별로 확보
- 검증 방식: 검색 요약이 아니라 실제 파일 저장소(S3 버킷 리스팅, GitHub/HuggingFace 트리)를 직접 열어 `.usd` 파일 존재를 확인. 로그인 없이 바로 접근 가능한 것만 포함. 전부 Isaac Sim에 실제로 스폰해서(`tools/02_robot_spawn.py`) 화면에 정상적으로 나오는 것까지 확인함.
- 제외 대상: 로봇팔/그리퍼/손만 있는 매니퓰레이터 자산, 드론, RL 벤치마크용 가상 캐릭터(Ant, Cartpole 등), 실제 하드웨어가 아닌 예제용 자산(BalanceBot, Vehicle, Leatherback, Simple 등)

## 요약

| 카테고리 | 확인된 종류 수 | 주 출처 |
|---|---|---|
| 바퀴형 | 14 | NVIDIA Isaac Sim Nucleus 공식 자산 |
| 다족보행 | 17 | NVIDIA Isaac Sim Nucleus + Unitree + Deep Robotics 공식 저장소 |
| 휴머노이드 | 18 | NVIDIA Isaac Sim Nucleus + Unitree + LimX Dynamics 공식 저장소 |
| **합계** | **49** | |

## 1. 바퀴형 (14종)

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

## 2. 다족보행 (17종)

| 로봇 | 제공처 | 출처 |
|---|---|---|
| A1, Go1, Go2, B2, Aliengo, Laikago | Unitree | Isaac Sim Nucleus `Robots/Unitree/{모델}/` |
| Go2W (바퀴+다리 하이브리드) | Unitree | 공식 GitHub/HuggingFace [`unitreerobotics/unitree_model`](https://huggingface.co/datasets/unitreerobotics/unitree_model) `Go2W/usd/` |
| ANYmal-B, ANYmal-C, ANYmal-D | ANYbotics | Isaac Sim Nucleus `Robots/ANYbotics/anymal_{b,c,d}/` |
| Spot (+with arm) | Boston Dynamics | Isaac Sim Nucleus `Robots/BostonDynamics/spot/` |
| Lite3, X30, M20, M20S, M20_Piper(+로봇팔), DR02(standard) | Deep Robotics | 공식 GitHub [`DeepRoboticsLab/deep_robotics_model`](https://github.com/DeepRoboticsLab/deep_robotics_model) |

**확인 결과 없는 것**: Ghost Robotics(Vision 60 등), Xiaomi CyberDog, LimX Dynamics 사족(W1/P1/CL-1, 공개 저장소에서 USD 확인 못함 - 휴머노이드/TRON2 계열만 USD 있음).

**제외함**: `Robots/NTNU/ARL-Robot-1/` — 실제 스폰 검증 과정에서 확인해보니 다족보행 로봇이 아니라 프로펠러 4개가 고정 관절(PhysicsFixedJoint)로 붙은 드론(쿼드콥터)이었음. 회전/직동 관절이 없어 관절 기반 로봇 검증에도 안 맞고, plan.md의 타겟 하드웨어(바퀴형/다족보행/휴머노이드)에도 해당하지 않아 목록에서 제외.

## 3. 휴머노이드 (18종)

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
| Tien Kung | XHumanoid | Isaac Sim Nucleus(5.1) `Robots/XHumanoid/Tien Kung/tienkung.usd` |
| PX5 | XiaoPeng | Isaac Sim Nucleus(5.1) `Robots/XiaoPeng/PX5/px5.usd` |
| A2D | Agibot | Isaac Sim Nucleus(5.1) `Robots/Agibot/A2D/A2D.usd` |
| T1 (locomotion) | Booster Robotics | Isaac Sim Nucleus(5.1) `Robots/BoosterRobotics/BoosterT1/T1_locomotion.usd` |
| Valkyrie | IHMC Robotics | Isaac Sim Nucleus(5.1) `Robots/Ihmcrobotics/Valkyrie/valkyrie.usd` |
| HU_D03, HU_D04(+그리퍼) | LimX Dynamics | 공식 GitHub [`limxdynamics/humanoid-description`](https://github.com/limxdynamics/humanoid-description) |
| TRON2A SF(하반신, 이족보행) | LimX Dynamics | 공식 GitHub [`limxdynamics/tron2-robot-description`](https://github.com/limxdynamics/tron2-robot-description) — 8개 변형 중 대표로 뽑음, 팔·상체 없이 다리만 있는 로코모션 베이스 |
| TRON2A WF(하반신, 바퀴-다리 하이브리드) | LimX Dynamics | 위와 동일 저장소 — 발 대신 바퀴가 달린 변형 |

**확인 결과 없는 것**: Tesla Optimus, PAL Talos, Ghost Robotics 계열, Booster K1/T2(T1만 확인), Unitree R1/H2 Plus, Galbot 휴머노이드 라인(`galbot_s1_description`은 지오메트리 페이로드만 있고 완성된 진입점 USD가 없어 제외), Agibot X2/X2Ultra(공개 저장소는 URDF만 확인됨).

## 방법론 메모

- NVIDIA Isaac Sim 자산은 `https://omniverse-content-production.s3-us-west-2.amazonaws.com/`가 로그인 없이 공개 리스팅되는 S3 버킷이라, `?list-type=2&prefix=...&delimiter=/` 쿼리로 폴더 구조와 실제 파일 존재 여부를 직접 확인할 수 있었다. 버전마다(4.5 vs 5.1) 포함된 로봇이 달라서 최소 두 버전을 함께 확인해야 누락이 없다.
- Unitree는 Isaac Nucleus에 없는 최신·변형 모델(H1-2, H2, Go2W)을 자체 GitHub/HuggingFace 저장소(`unitreerobotics/unitree_model`)에 별도로 공개하고 있어, 두 소스를 모두 확인해야 한다.
- Deep Robotics·LimX Dynamics처럼 회사가 자체 GitHub에 `usd/` 폴더를 직접 커밋해 공개하는 경우가 있다 - Isaac Nucleus에 없다고 "USD 공개 안 됨"으로 단정하면 안 되고, 회사 GitHub 계정을 따로 확인해야 한다(초기 조사에서 Deep Robotics·LimX를 "확인 결과 없음"으로 잘못 적었던 이유).
- "USD 파일이 저장소에 존재한다"와 "Isaac Sim에서 바로 스폰해서 정상 작동한다"는 별개 문제다. 특히 크기가 비정상적으로 작은 파일(H2의 1.45KB 등)은 참조용 스텁일 가능성이 있어 실제 사용 전 열어서 확인이 필요하다.
- **텍스처(색) 유무는 출처에 따라 갈린다.** NVIDIA가 직접 공들인 "플래그십" Nucleus 자산(Aliengo, Spot, ANYmal, GR-1, Valkyrie, Nova Carter 등)은 사진 같은 PBR 텍스처가 있지만, 회사가 CAD/STL을 자동 변환 도구로 그대로 내보낸 경우(Deep Robotics, LimX, Turtlebot3, iwhub 등)는 재질이 `diffuse_color_constant = (1,1,1)`(흰색) 하나로 통일돼 있다 - STL 자체가 색 정보를 담지 못하는 포맷이라 변환 과정에서 색이 통째로 유실된 것으로, 파이프라인 버그가 아니라 원본 데이터의 한계다. 형상·관절 검증에는 영향 없음.
