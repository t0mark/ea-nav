# EA-Trav 작업 기록

## 컨테이너 구성 (docker/)

| 컨테이너 | 용도 | 비고 |
| --- | --- | --- |
| airlab_hw_eatrav_sim | Isaac Lab 2.3.0 시뮬 | 기존 수동 생성, compose는 문서화용 |
| airlab_hw_eatrav_train | PyTorch 학습·데이터 처리 | nvcr pytorch:24.02 |
| airlab_hw_eatrav_utils | 기타 CPU 작업 (URDF 생성·검증·수집) | python:3.10-slim, GPU 미할당 |

- 마운트 공통: `../ -> /workspace/eatrav`, `~/jairlab/data -> /data`
- Dockerfile은 접미사 방식 명명: `Dockerfile.train`, `Dockerfile.utils` (compose 공통 설정은 x-common/x-gpu 앵커)
- utils 의존성은 Dockerfile.utils에 반영 완료 (numpy, scipy, trimesh, python-fcl, yourdfpy, xacro, robot_descriptions 등)
- sim 컨테이너 캐시 마운트(/data/EA-Trav/isaac-cache/...) 주의: 호스트 쪽 폴더를 지우면 고아 마운트가 되어
  셰이더·텍스처 캐시 쓰기가 전부 실패한다 -> 폴더 재생성 후 컨테이너 재시작으로 복구

## 1단계: URDF 랜덤 생성기 (scripts/urdf/) — plan/urdf.md 확정본 기준 전면 재구현

### 패키지 구조와 파일별 요약

설계 기준: 호출 사이에 유지되는 상태가 있으면 클래스, 없으면 모듈 함수 (AGENT.md 규칙).
표기 랜덤화는 plan 확정대로 인코더 입력측 증강(4단계 파서 소관)이라 이 단계에는 없음 — 산출물은 정본 URDF뿐.

- `core/base.py` : 공통 기반
  - 스펙 자료구조: GeomSpec / LinkSpec / JointSpec / RobotSpec (+ check_poses 추가 검사 자세, special 특수 검사 선언)
  - 관성 계산 함수 `compute_com/compute_inertia` (평행축 정리), URDF 쓰기 함수 `write_urdf`
  - BaseGenerator 추상 클래스 (전체 cfg 보관, `_u` 범위 샘플, `_clamp_mass_ratio` 질량비 500 클램프)
- `platform/wheeled_base.py` : wheeled 공통 부품 (WheeledBase)
  - 몸통 형상 축(box/cylinder/stack/cylinder_stack), 바퀴(자식 프레임 회전축 +y 통일, joint_yaw로 방사 배치),
    롤러(틸트 파라미터: ±45도 매커넘 / 0도 옴니휠), 트랙 계산(노출/은닉), 무게중심 오프셋, 구동 한계(`_set_drive_limits` — 상체 추가 후 재호출 가능)
- `platform/diff.py` : DiffGenerator. 캐스터 종류 축 = 없음(2륜 balancing)/볼/스위블, 배치 축 = 반대편1/양쪽/반대편2
  - balancing: 몸통 키움 + 무게중심을 축 위·축선상에 구성적 배치, special["balancing"] 등록, control_tag = diff_balancing
  - 하중 비율: com_x를 지렛대 비율에서 역산해 구성적 샘플 + special["load_share"] 이중 확인
  - allow_balancing=False 인자 (wheeled 휴머노이드 상속용)
- `platform/skid.py` : SkidGenerator. 4/6륜 축, 바퀴 지름 < 최소 축간격 클램프, 휠베이스:트랙 비율 상한 1.5
- `platform/ackermann.py` : AckermannGenerator. 구동 방식 축(후륜/전륜/4륜 — 수동 휠 effort 0), 킹핀 오프셋 유무,
  앞 트랙을 조향 침범량 D=(ko-w/2)(1-cos)+r·sin 만큼 확장, 조향 극한 자세 검사, 최소 회전 반경 = wb/tan(δ) ≤ 4L,
  조향 너클 토크·속도 샘플, 구동축 하중 비율 구성적 배치
- `platform/omni.py` : OmniGenerator. 하위 타입 축 = mecanum4 / omni3 / omni4
  - 매커넘 표준 X 고정: 롤러 축 y부호 s = -sign(x*y) -> yaw 모멘트 팔 = |x|+|y| (유도 주석·URDF 기반 검산 완료)
  - 롤러 길이 = 커버리지 역산, 트랙 = 롤러 안쪽 돌출량만큼 확장, 옴니휠은 방사 배치(joint_yaw) + 접선 롤러
- `platform/multileg.py` : MultilegGenerator (quad/hex). 장착 mammal/sprawl, 무릎 방향 열별 독립, 2/3절,
  고관절 축 순서 = roll-pitch/pitch-roll(mammal), yaw-pitch(sprawl = coxa 구조), 기립 IK(법코사인) + 다리 자동 보정
- `platform/humanoid.py` : HumanoidGenerator. 상체 분할(merged/waist), 발 형상(box/disc/dual = 링크 1개 + 충돌 2개),
  팔 0/3-7 DoF (어깨3 + 팔꿈치 + 손목3 체인 앞에서 자름), 고관절 축 순서 순열, 무릎 굽힘 2링크 IK + 발바닥 수평
- `platform/wheeled_humanoid.py` : WheeledHumanoidGenerator. 베이스(diff/skid/omni, balancing 제외) 재사용 +
  몸통(리프트 prismatic 50% / fixed) + 팔 1-2개 x 2-7 DoF + 머리. special["overturn"], 리프트 최대·팔 전방 추가 자세,
  상체 반영 구동 한계 재설정. 롤아웃 명목 자세 = 리프트 최하단 + 팔 홈 포즈 (plan GT 정의)
- `utils/loader.py` : PosedModel (yourdfpy FK -> 링크별 월드 trimesh·질량중심. 검사·렌더 공용)
- `utils/validate.py` : `validate_static()`. 공통 5검사 = 지상고 / 셀프 충돌(거리1 + 가동 형제만 허용) /
  무게중심-지지다각형 / 중력 토크(prismatic은 축력) / 접지 공면(롤러는 바퀴 단위 그룹) + 추가 자세 셀프 충돌
  + 특수 검사 3종 = load_share(1D 지렛대) / balancing(축선·축위·복원토크) / overturn(무게중심높이/경계여유 ≤ 8)
- `utils/render.py` : `render_robot()` (matplotlib 헤드리스, 단일 컬렉션 깊이 정렬)
- `pipeline.py` : GenerationPipeline. 시드 = (전역, form 인덱스, 로봇 인덱스, 시도) -> 규모 확장 시 앞 인덱스 재현.
  form 순서 = diff, skid, ackermann, omni, quad, hex, humanoid, wheeled_humanoid (8종)
- `real_robots.py` : RealRobotCollector / RealPoolReport (mimic 제외 독립 DoF 집계 포함)

### 설정·실행

- 설정: `configs/urdf.yaml` (파라미터 범위·검사 임계값. 범위 원칙 = 실로봇 풀 질량 1-265kg 지지범위 포괄)
- 진입점: `tools/00_urdf.py` — generate --mode pilot|full --count N [--seed --workers] / validate --mode /
  collect-real / pool-report [--mode]. 파라미터 설명은 파일 상단 docstring
- 진입점 공통 유틸: `tools/utils/common.py` (저장소·데이터 루트 상수, config 로드, 로깅 초기화 — sim 계열 진입점과 공유)
  - 렌더 루프 `render_set`·재검증 루프 `validate_set`은 scripts/urdf/utils 쪽 라이브러리 함수
- 경로 고정 (전부 파일 기준 상대 계산, 실행 cwd 무관): 파일럿 = check/00_urdf/{robots,renders}(렌더 자동),
  본 생성 = /data/EA-Trav/urdf/synthesis, 풀 = /data/EA-Trav/urdf/real_robots
- 실행: utils 컨테이너에서 `python /workspace/eatrav/tools/00_urdf.py ...`
- 산출물: `{루트}/{form}/{form}_{idx:04d}/robot.urdf + meta.json`
  (meta = params 속성 딕셔너리, standing_pose, contact_links, control_tag, special_checks, metrics.base_height = 스폰 높이)

### 플랜-코드 정합성 검증 (에이전트 2라운드, 합격)

- 대조 결과 약 70여 항목 일치. 발견·수정 4건: balancing 검사 좌표 버그(base 프레임 com을 지면 기준 축 높이와 비교 -> 오기각 95%. lever = (com_z - z0) - axle_z 로 수정, 통과율 80%로 정상화) / 조향 너클 토크를 전축 하중 연동으로 교체 / 추가 자세(팔 전방·리프트 최대)에서 com-다각형·전복 재검사 추가 / 실로봇 풀 보수(spot 관성 32.5kg 주입, mesh AABB로 bbox 산출 — trimesh+pycollada)
- 방어 수정: 반지름 0 구 등 퇴화 프리미티브는 로더에서 건너뜀 (atlas_drc 발의 r=0 구가 NaN 정점 유발)
- 플랜을 코드에 맞춘 것 2건: omni 몸통 형상에 적층 추가(+config cylinder), 전복 안정성 정의를 "경계 여유(최악 전도축)" 기준으로 명확화

### 파일럿·검증 결과

- 파일럿 8 form x 3 = 24/24 통과 (추가 자세 검사 포함), 재검증 24/24, 렌더 24장
- 대량 검산 (form당 25, 시드 7): 전 form 실패 0 (평균 시도 1.0-2.6)
  - 매커넘 yaw 모멘트 팔 = |x|+|y| 검산 7/7 정상 (URDF에서 롤러 축 재추출로 확인)
  - ackermann 조향 전 범위 스윕 충돌 0/25, 접지 공면 산포 최대 3.9mm (허용 8mm)
  - diff 서브타입 분포 swivel/ball/balancing = 9/14/2, balancing 성립 조건 전수 확인
  - wheeled_humanoid: 베이스 diff/skid/omni 골고루, 리프트(prismatic) 10/25
  - 커버리지: 분포 지지범위가 풀 전체 포괄 (배치 꼬리값은 대량 생성에서 채워짐)

## 실로봇 URDF 풀 (/data/EA-Trav/urdf/real_robots/) — 40대, 파서 40/40 통과

- multileg 15: a1, aliengo, anymal_b/c/d, b1, b2, go1, go2, hyq, laikago, mini_cheetah, solo, spot, **phantomx(6족)**
- humanoid 13: g1, h1, atlas_drc/v4, talos, jvrc, icub, valkyrie, jaxon, sigmaban, berkeley_humanoid, draco3, ergocub
- wheeled 12: turtlebot3 burger/waffle, husky, jackal, dingo, upkie, stretch, fetch, pr2, tiago, **ridgeback(매커넘)**, **racecar(ackermann)**
- plan 보강 반영: ackermann·omni·6족 실물 확보 -> 해당 form 제로샷 평가 가능. youbot은 구식 xacro 내부 변수 문제로 수집 실패 (로그 기록, 매커넘 대표는 ridgeback)
- 수집: robot_descriptions + 저장소 클론 (xacro는 $(find) 제자리 치환 + M_PI->pi 호환 처리 + 외부 센서 include 제거 후 API 변환)

### 특이사항 (나중에 고려)

- 동적 검사(평지 기립+전진, balancing은 직립 스폰 유지+전진)는 2단계 제어기 이후 별도 필터
- 가동 범위 내 임의 자세 셀프 충돌은 허용 (실로봇 동일, 시뮬 접촉으로 해소. 기립·조향 극한·리프트 최대·팔 전방만 필터)
- 매커넘 원통 롤러 근사로 유효 반지름 1-2% 진동 잔존, 실로봇 매커넘은 URDF상 skid와 구별 불가 (plan 감수 — 평가 분리)
- 표기 랜덤화(형상 근사·관성 노이즈·더미 센서·mesh->AABB)는 4단계 파서의 입력측 증강으로 구현 (이 단계 산출물은 정본)
- mimic 조인트: 생성 URDF에 없음. 실로봇 풀(pr2/talos/jvrc 그리퍼)은 yourdfpy가 종속 DoF로 정상 처리, 4단계 파서에서 fixed처럼 병합 권장
- spot URDF는 원본에 관성 없음(질량 0) -> 제로샷 평가 전 관성 보수 필요. mesh 전용 로봇 bbox는 mesh 로드 기반 산출 필요
- 커버리지 극단값(1kg 미만, 250kg 초과)은 저확률 꼬리라 대량 생성에서 소수만 등장 (지지범위는 포괄)

## sim 기반 (scripts/sim/utils/ + tools/01_sim.py) — URDF->USD·검증·스폰·지형·렌더

controller(2단계)·trav_gt(3단계)는 sim 의존이라 scripts/sim/ 하위에 배치 예정 (별도 세션에서 작업).

### H200(RTX 코어 없음) 실행법 — 실측 확인

- 표준 Isaac Lab AppLauncher(headless=True + enable_cameras=True)로 그대로 기동·렌더 동작.
  별도 우회 플래그·수정 파일 없음 (드라이버 590.48.01, Vulkan 레이트레이싱 경로. H200 4장 인식)
- 오프스크린 이미지는 **Camera 센서(render product)** 경로만 동작.
  뷰포트 캡처(capture_viewport_to_file)는 창 없는 헤드리스에서 파일을 쓰지 못함 (실측 — 앱은 정상, 파일만 안 생김)
- 실행: sim 컨테이너에서 `/isaac-sim/python.sh /workspace/eatrav/tools/01_sim.py --mode pilot|full`
- 렌더 비활성(enable_cameras=False) 기동이 더 가벼움 -> launch_app이 pilot 모드에서만 렌더 활성

### 파일별 요약

- `tools/utils/sim.py` : 진입점 공통 부트스트랩 (01·02·03 공유)
  - `launch_app(parser, enable_cameras)` : 헤드리스 Isaac 앱 기동. enable_cameras = True/False/"pilot"
    (--mode pilot일 때만). 앱이 뜬 뒤에야 Isaac 의존 모듈(scripts.sim.*, pxr, omni.*)을 임포트할 수 있다
  - `close_app(app)` : STOP 콜백 무한 렌더 루프 방지 플래그를 켜고 종료 (app.close() 직접 호출 금지 —
    프로세스가 코어 점유 상태로 안 죽는 것 실측)
- `scripts/sim/utils/robot_spawn.py` : 로봇 에셋 준비 전부 (상태 없는 모듈 함수)
  - `parse_joints/classify_drive_groups/compute_drive_gains` : 드라이브 분류 = effort<=0 passive /
    continuous velocity(바퀴) / 나머지 position. 변환 검증·스폰이 같은 규칙 공유 (configs/sim.yaml drive와 짝)
  - `convert_robot` : URDF -> USD 1대 변환 (게인 중립 — USD에 안 굽고 스폰 시점에 부여) + meta.json 복사 +
    joints.json 동봉 (하위 단계가 URDF 재파싱 없이 동작)
  - `inspect_usd` : USD를 열어 가동 조인트 보존·게인 0·articulation root 대조
  - `make_articulation_cfg` : 스폰 스펙(초기 기립 자세 + 스폰 높이 + 조인트별 게인) 구성
  - `iter_robot_dirs` : {루트}/{form}/{이름}/ 2단 구조 순회
- `scripts/sim/utils/environment.py` : `SimEnvironment` 클래스 — 시뮬 환경 1개의 수명 관리.
  생성(이전 환경 자동 정리) -> add_ground(평지) 또는 add_terrain(TerrainGenerator) -> spawn_robot(GridCloner
  단일/다중 env, color = 렌더용 visual_material, self_collision 기본 켬) -> reset() -> step(목표)/hold_step(기립 홀드)
  - 지면 경로는 평지·지형 공통 /World/ground (RL 원본과 동일 — 2단계 height_scanner 경로 그대로 사용 가능)
  - 셀프충돌은 게인과 동일 원칙으로 USD에 굽지 않고 스폰 시점 articulation_props로 활성.
    1단계 정적 검사가 "임의 자세 셀프충돌은 시뮬 접촉으로 해소" 전제로 허용했으므로 켜는 게 정합
    (configs sim_run.self_collision)
  - add_terrain: Isaac Lab 보행 RL 표준(ROUGH_TERRAINS_CFG, velocity_env_cfg)과 정합 — 서브지형 6종 =
    피라미드 계단/역계단/랜덤 박스/랜덤 러프/피라미드 경사/역경사 (종류·비율·치수 RL 원본과 동일 값,
    GitHub 대조 확인) + 난이도 커리큘럼(행 방향) + 지면 마찰(multiply 1.0/1.0) 명시.
    격자 규모만 config로 축소 (파일럿 4x4, RL 원본 10x20 — 열이 적으면 비율 0.1짜리 경사 2종은 배정 안 됨).
    반환 = 전체 크기 (탑뷰 카메라 고도용). 로봇 스폰은 현재 평지 기준, 지형 위 스폰은 3단계에서
    max_init_terrain_level·지형 원점과 함께
- `scripts/sim/utils/render.py` : 정지 캡처 + 주행 비디오 + 씬 장식. reset 전에 카메라 생성.
  - `SceneCamera` : Camera 센서 캡처 — `capture_array`(RGB 배열)·`capture_rgb`(PNG 저장)
  - `VideoRecorder` : 프레임 누적 -> H.264(yuv420p) mp4 인코딩 (imageio_ffmpeg 번들
    ffmpeg — VSCode 내장 미리보기 재생 확인. 연속 캡처는 워밍업 2프레임으로 충분)
  - 씬 장식 (파일럿 시각 확인 전용, 순수 시각 프림 — 재생 중 생성 무해):
    `spawn_goal_markers`(웨이포인트 기둥+구, 순서색) / `spawn_breadcrumb`(궤적 표식) /
    `add_ground_grid`(평지 격자선 — 단색 지면의 이동 기준선)
- `tools/01_sim.py` : 진입점. --mode pilot = check 셋 변환+검증+지형 탑뷰+스폰+기립 홀드 시뮬+form 색 렌더,
  --mode full = /data 본 셋 변환+검증+시뮬 실행 확인(렌더 없음). 판정은 NaN/발산 여부만 기록
  (자세 평가는 제어기 단계 소관 — 제어기 없는 PD 홀드는 humanoid가 넘어지는 게 정상)
  - 개체별 예외 격리: 변환·검증·시뮬 세 루프 모두 한 대 실패가 셋 처리를 끊지 않음 (실패는 report에 error로 기록,
    보고 저장·close_app 항상 도달). 시뮬은 변환+검증을 모두 통과한 개체만 진입
  - 단건 테스트: `--robot {form}/{이름}` = 로봇 1대만 (시점·색 확인), `--terrain-only` = 지형+탑뷰만 (pilot 전용, 가드 있음).
    둘 다 보고 json을 저장하지 않음 — 시점 보정 때 전체 재실행 금지, 단건으로 확인
- `configs/sim.yaml` : converter(변환 옵션) / drive(게인 규칙 k = scale x effort) / sim_run(물리 스텝·실행 시간) /
  terrain(서브지형 비율·치수) / render(해상도·카메라 거리 = 2.2 x 최대 외형 치수·form_colors)
- 경로: pilot = check/00_urdf/robots -> check/01_sim/{usd,renders,terrain_topview.png,report_pilot.json},
  full = /data/EA-Trav/urdf/synthesis -> /data/EA-Trav/sim/{usd,report_full.json}

### Isaac Lab 2.3.0 실측 특성 (코드에 반영)

- 타임라인 STOP 콜백이 헤드리스에서 무한 렌더 루프 -> stop 직전에 `_disable_app_control_on_stop_handle=True`
  (reset()이 매번 False로 되돌리므로 생성 시점 설정은 무효 — _teardown_previous와 close_app 양쪽에 반영)
- GroundPlaneCfg는 Nucleus 클라우드 에셋 참조라 오프라인에서 실패 -> `physicsUtils.add_ground_plane` 프로시저럴 평면
- TerrainImporterCfg는 visual_material 기본값(검정 PreviewSurface, 로컬)인데도 이 환경에서 CPU 0% 무한 대기 실측
  (원인 미규명) -> visual_material=None + 높이 색상(color_scheme height)으로 회피. 지형 생성 자체는 로컬 trimesh 연산 2초 수준
- 스포너는 프림만 배치하고 물리 상태를 안 씀 -> reset()에서 `write_joint_state_to_sim` 등 명시 기록
  (빠지면 legged가 영점 자세에서 시작해 전도)
- 스테이지 프림 직접 수정(displayColor)이 로봇 렌더에 반영되지 않는 것 실측 (make_instanceable=False인데도
  스테이지 순회에서 Gprim 미검출 — 원인 미규명) -> 로봇 색은 스폰 시점 UsdFileCfg visual_material로 입힌다
- `merge_fixed_joints=True`여도 질량·충돌 있는 fixed 링크는 병합 안 됨 (PhysicsFixedJoint 유지 — 물리 동일, 검사 허용)
- URDF effort 0은 드라이브 maxForce 무제한으로 임포트 -> 수동성은 게인 0으로 보장 (검사도 게인 기준)
- 카메라 API: 시점 지정은 `Camera.set_world_poses_from_view(eyes, targets)` (set_world_poses는 시그니처 다름).
  수직 탑뷰는 -y 미세 오프셋으로 업벡터 퇴화 방지 + 지도 방향 정렬
- `app.close()`는 fastShutdown으로 프로세스를 즉시 끝냄 -> close 이후 코드(SystemExit 메시지 등)는 실행 안 됨.
  오류 사유는 close 전에 logger로 남길 것 (01_sim 가드 패턴 — 02·03 진입점도 동일하게)
- 스폰 로그의 "셀프충돌" 값은 요청값이 아니라 스테이지 실값(physxArticulation:enabledSelfCollisions)을 읽은 것
  — articulation_props가 조용히 무시되는 실패를 로그에서 바로 드러내기 위함 (None = 적용 실패)

### 파일럿 결과 (check/01_sim/, 24대 = 8 form x 3)

- 변환 24/24, 정합성 검증 24/24, 시뮬 실행(4초 기립 홀드) NaN·발산 0건, form 색 렌더 24장 + 지형 탑뷰 1장
- 지형: 4x4칸 36x36m, RL 정합 서브지형 생성·탑뷰 판독 정상 (계단 = 빨강, 역계단 = 파랑(음수 높이),
  행 내 좌->우 난이도 상승 = 커리큘럼 확인)
- 제어기 없는 홀드의 관찰값: humanoid_0000 전도(tilt 90도), diff_0002(balancing) 기울어짐 24.5도 — 예상 거동,
  2단계에서 RL 정책·LQR이 담당. 나머지 22대는 기립 유지 (tilt < 7도)
- 처리 시간: 전체 파일럿 약 4분 (셰이더 캐시 워밍업 후. 첫 기동은 캐시 컴파일로 수 분 추가)

### 특이사항 (나중에 고려)

- plan의 오버행/통로 구조물(책상·낮은 천장·좁은 벽)은 Isaac 기본 서브지형에 없음 -> 3단계에서 직접 메시 제작
- 볼 캐스터는 fixed 병합이라 회전 없는 미끄럼 접촉 -> 롤아웃 전에 캐스터·롤러 공통 마찰 상수 설정 필요
  (plan 제어기 섹션의 상수화 원칙)
- 홀드 게인 규칙 k=10x effort는 실로봇 관례(Go2 k=25) 대비 과강성 의심 — 제어기 단계에서 홈 포즈 홀드 게인 재검토
- full 모드(본 생성 셋)는 00_urdf full 생성 후 실행 (/data/EA-Trav/urdf/synthesis 현재 없음)
- 프로세스 1개에서 SimEnvironment 재생성 24회 반복 정상 (스테이지 누수 없음)
- 지형 시드는 현재 미고정 (TerrainGenerator 전역 난수) — 3단계 GT 생성 때 시드 체계에 편입 필요

## 2단계: wheeled 제어기 (scripts/sim/controller/) — pure pursuit 명령 + IK/LQR 배분 캐스케이드

구조: 상위 = pure pursuit(결정적 기하 제어, 목표점 -> 몸체 속도 명령), 하위 =
역기구학(일반 wheeled) 또는 LQR 균형 토크(diff_balancing). legged는 아래 별도 섹션.

처음엔 MPPI(샘플링 MPC)로 구현·15/15 완주까지 갔으나, 장애물 회피가 없는 이
연구의 사용 형태에서 샘플링의 대가(속도 활용 25-35%, 잔여 사행, 시드 관리)만
남아 pure pursuit으로 교체 (사용자 결정). 교체 후 결정적·재현적이며 도달 시간
30-50% 단축. MPPI 시절 교훈은 아래 "실측 특성"에 보존.

### 파일별 요약

- `core/base.py` : 공통 기반 (Isaac 비의존 — 앱 기동 전 임포트 가능)
  - BaseController(ABC, reset(env_ids) 부분 리셋 계약) + ControlObs(쿼터니언 -> yaw·pitch, 몸체 속도)
    + JointTargets(전 DoF pos/vel/effort 텐서 + cmd = 모델 수준 몸체 명령 — 3단계 WVN 추종 점수용)
  - URDF 파싱·조인트 0 자세 FK(`zero_pose_frame`) -> 구동 바퀴 배치 추출 `extract_ctrl_params`
    (meta.json 대표 치수 + joints.json 한계 + robot.urdf 기하·관성 조합, tip_accel 기하 유도 포함)
  - `make_controller` 팩토리: control_tag 분기, wheeled_humanoid_* 접두사는 베이스 타입으로 환원
- `core/high_pp.py` : PurePursuit (torch 배치, 결정적 — 난수 없음).
  wheeled/에서 core/로 이동 — legged도 같은 명령 계층을 쓴다 (plan "공통 명령 계층")
  - 구간 기준선(목표 변경 시점 위치 -> 목표) 위 전방 주시점 추종. 곡률 k = 2 sin(alpha)/L_d
    (명목 주시 거리 기준 표준식 — 실거리 제곱으로 나누면 이탈할수록 조향이 약해져
    목표 중심 광궤도 공전이 생기는 것 실측)
  - 속도 프로파일 = min(상한, 정지 sqrt(2 a d), 곡률 감속 sqrt(a_lat/|k|)) — 목표·코너 앞 자동 감속
  - 모델 3종: unicycle(diff/skid/mecanum — 방위 오차 크면 선회 우선 + 저속 전진 creep) /
    bicycle(ackermann — 목표가 뒤·최소 회전원 안이면 조향 후진 재정렬, 방위각 히스테리시스
    래치: 즉시 판정은 옆 목표에서 전/후진 채터링 실측) / holonomic(omni3·4 — 주시점 방향
    속도 벡터 + 진행 방향 정렬 P)
- `wheeled/low_ik.py` : 배분 행렬 A(바퀴 수 x 3) 로봇당 1회 구성 — 일반 바퀴 행 = 굴림 방향 t = axis x z,
  매커넘 행 = 롤러 구속 s = -sign(x*y) (+y 축 부호 검증), 옴니 방사는 일반 행으로 자동 처리.
  feasible_scale은 (v,vy,w) 통째 축소로 경로 형상 보존. ackermann_steer = 애커먼 좌우 배분 (ICR 기하)
- `wheeled/low_lqr.py` : URDF 링크 관성 합성(회전 + 평행축) -> 2륜 역진자 선형화 (유도 주석) -> CARE 게인
  로봇당 1회 -> 배치 토크. yaw는 P 차동 토크 (토크 한계 비례 게인 — 질량 1-265kg 규모 정합).
  평형 피치 오프셋(질량중심 x 잔차) 보정 포함
- `wheeled/controller.py` : 조립 2종 + 실측 보정 로직
  - WheeledRobotController: PP + IK. 명령 경계 = 물성 유도 + yaw 절대 캡 + 전도 한계(tip_accel
    안전율 -> 가감속·원심·곡률 감속 상한). 슬루(가감속) 제한. 애커먼 중심각 상한 = 내륜 한계
    역산 (kappa_max = tanL/(wb + y_in tanL)), 전륜구동 바퀴 1/cos(조향각) 보정.
    적응형 yaw 보상(diff·skid + 전도 여유 충분한 mecanum): 명령 대비 실측 yaw 비율로 차동 배율
    온라인 갱신 + 실현 횡가속 클램프 |v x w| <= lat_accel. make_controller의 필수 인자 on_terrain이
    지형에서 자동 차단 (기본값 없음 — 누락 시 TypeError 인터록).
    diff 최소 선회 반경 = 1.5 x 트랙 (아래 스틱-슬립 실측)
  - BalancingRobotController: LQR 매 물리 스텝(200Hz) + PP 명령 갱신만 decimation 주기.
    바퀴 게인 0 스폰 (gain override) + effort 목표 전제. 플래너 감속 가정 = brake_ratio 별도 하향
    (역진자 비최소위상 — 아래 특이사항)
- `tools/02_controller.py` : 진입점 (wheeled·legged 공용, 서브커맨드 eval|train —
  기존 CLI의 시나리오 평가는 eval로 편입). eval --mode pilot(check 셋 x 씬:
  wheeled = 평지·경사, legged = 평지·경사·계단 — 씬별 궤적 플롯 + 주행 비디오) |
  full(본 셋 동적 검사 = plan 정의대로 평지만, 비디오 없음), --robot 단건.
  경사·계단 씬 = 기존 add_terrain 재사용 (해당 서브지형 비율만 1인 1x1 역피라미드
  — 중앙 스폰에서 바깥으로 등판, 난이도는 범위 최소=최대 고정. controller.yaml
  eval_terrain). 지형 씬은 make_controller(on_terrain=True) — wheeled 적응 보상 차단.
  시나리오 = 직진 -> 좌회전 -> 우회전 (balancing = 직립 유지 -> 직진 -> 선회),
  제한 시간 = 물성 비례 (nav_limits 실경계 기준). 도달 판정 = 반경 진입 + 0.3s 체류 + 서행
  (속도 < 0.5 m/s — 체류만으로는 2.33 m/s 미만의 무정지 관통을 못 거르는 것 감사 반증.
  0.3은 balancing 정지 유지 진동에 걸리는 것 실측 -> 0.5)
- `configs/controller.yaml` : ctrl(주기·경계·슬루·보상) / pp(주시·감속·선회·후진) / lqr / scenario

### 기존 코드 확장

- `environment.step`에 joint_effort_target 인자 (LQR 토크), `spawn_robot`에 gain_overrides(조인트별 게인
  덮어쓰기)·contact_cfg(수동 접촉 마찰 바인딩 — 복제 전 env_0에 적용) 추가, 스폰 시 PhysX
  sleep/stabilization threshold 0 (저속 절전 방지 보수 설정)
- sim.yaml `contact` 섹션: 볼 캐스터 mu 0.01(이상적 롤링 근사), 롤러 mu 1.0 (URDF에 없는 물리의 공통 상수화)
- sim.yaml `vel_damping_scale` 1.0 -> 3.0 (접촉 부하 시 바퀴 목표 미추종 실측)

### 파일럿 결과 (check/02_controller/wheeled/, 15대 = wheeled 5 form x 3)

- 15/15 완주 (반경 0.35m + 0.3s 체류), 도달 시간 8.7-25.8s, 기울기 최대 1.6도 (전도 0)
- 궤적: 직선-원호-직선의 정상 주행. ackermann은 정당한 후진 재정렬만 잔존, 사행·공전 없음
- 결정적 제어라 같은 조건 재실행 = 같은 궤적 (GT 재현성 — 시드 관리 불필요)
- 산출물: 궤적 플롯 15장(plots/) + 렌더 15장(renders/) + report_pilot.json

### 독립 감사 요약 (에이전트: MPPI기 3회 + PP 교체본 2라운드, 전부 합의 성립)

PP 교체본 감사 (2라운드 합의): 수식 전량 재유도 결함 0 (PP 곡률·속도 프로파일·애커먼
배분/내륜 역산·역진자·매커넘 배분·전륜 cos 보정 전부 일치), 도달 시간 수치 역산 정합.
지적 -> 수정: 도달 판정 서행 조건, on_terrain 필수 인자 인터록, lookahead 회전 반경 비례,
플랜·코드 문서 정리, goal_change_eps 등 config화, holonomic 속도 크기, 지면 마찰 명시(아래).
지면 마찰 정합(0.5 기본 -> 1.0 명시)이 드러낸 후속 실측 3건도 해소: 적응 게인의 실현 횡가속
클램프 / mecanum 보상은 전도 여유(tip_accel >= 5) 조건부 (요동형 yaw 지터 증폭이 상체 무거운
개체를 전도시키고, 전면 제외하면 저상 mecanum이 선회 권한 부족 — 양쪽 실측) / 서행 문턱 0.5.

(MPPI기 감사 기록 — 교체로 코드가 사라졌으나 판정·교훈은 유효)

- 정합성 감사: 핵심 수식(매커넘 배분·애커먼 배분·역진자 선형화·CARE) 재유도 대조 결함 없음.
  지적 -> 수정: 다중 env 부분 리셋, 애커먼 중심각-내륜 정합, 전륜 cos 보정, 적응 보상 기준 교정
  (feasible_scale 후 값 — 양의 되먹임 차단), cmd 노출, 실경계 기반 제한 시간, 상수 config화
- 궤적 품질 감사: 사행(횡이탈 무구속)·전속 관통·확률적 비재현성이 좁은 통로 GT를 오염시킬
  경로로 판정 -> 경로 기준선 추종(PP 교체로 구조 해결)·체류 판정·결정적 제어로 전부 차단

### Isaac Lab 2.3.0 실측 특성 (제어 관련 — 코드에 반영)

- **물리 재질 바인딩 함정**: 변환 USD의 콜라이더는 인스턴스 프록시라 링크에 CollisionAPI가 없고,
  Isaac bind_physics_material은 조용히 False 반환 -> 마찰 상수가 전혀 적용되지 않은 채
  PhysX 기본(0.5)으로 돌던 것 실측. UsdShade 물리 바인딩을 링크 프림에 직접 걸어 해결
  (상위 바인딩은 프록시 콜라이더에 상속). 콜라이더 순회는 Usd.TraverseInstanceProxies 필요
- **볼 캐스터 마찰이 차동 선회를 지배**: 기본 마찰(0.5)에서 캐스터 하중 비중이 큰 diff는
  좌우 바퀴가 같은 속도로 강제되어 선회 불능 (바인딩 수정 + mu 0.01로 해결)
- **diff 협트랙 스틱-슬립 교착**: 전진 중 안쪽 바퀴가 정지·역회전 근방이면 하중이 빠지며
  구동 바퀴 헛돎 + 차체 동결 -> 최소 선회 반경 1.5 x 트랙 클램프 (역회전은 제자리 선회 전용)
- **mecanum4 횡이동 불성립 (바인딩 수정 후 재검증 확정)**: 45도 원통 롤러가 횡접촉력을 못 만듦
  (dt 1/4·솔버 64·유효 마찰에도 동일) -> 비홀로노믹 강등 유지. omni3(틸트 0 롤러)는 정상.
  GT·제로샷에서 mecanum은 skid처럼 거동 전제 (plan 감수와 정합)
- skid는 정지 제자리 회전이 마찰에 잠김 (주행 중 선회만) -> 선회 우선 모드에 저속 전진 creep
  (협트랙 diff도 동일 적용). 옆미끄럼은 정적 배율 2.2 + 적응 보상
- 명령 급가감속 저크는 상체 무거운 개체를 전도시킴 -> 슬루 + 기하 유도 tip_accel 상한
- MPPI 시절 교훈 (교체로 소멸했지만 기록): 웜스타트 시프트는 경과 시간 정합 필수 / 가우시안
  탐색만으로는 감속·후진 해 희소 / 호라이즌 < 선회 특성 시간이면 목표 옆 주차 국소해 /
  스무딩은 평균이 아닌 노이즈에 / 점 목표 거리 비용만으로는 사행이 공짜

### 특이사항 (나중에 고려)

- 3단계 롤아웃은 make_controller + ControlObs + JointTargets 인터페이스 재사용. 전 연산 배치
  설계이나 파일럿은 num_envs=1만 실측 — 다중 env는 3단계 초기에 확인. 셀별 에피소드 종료 시
  reset(env_ids) 부분 리셋 필수
- 제어기가 결정적이므로 3단계 반복 시도의 다양화는 스폰 자세 지터(위치·yaw 섭동)로 만들 것
  (같은 스폰이면 같은 궤적)
- 3단계의 적응형 yaw 보상 차단은 make_controller(on_terrain=True)가 자동 수행 (필수 인자 인터록)
- full 실행 전 단건 확인: 크기 비스케일 상수(stop_dist·reach_radius)를 양 극단 개체(최대 r_turn +
  최소 전장)로 검증할 것 (소형에서 0.35m 반경은 과관대 방향). mecanum_adapt_min_tip 5.0은 파일럿
  3대 분포 기반 — tip 4-6 경계 mecanum이 full 동적 검사에서 탈락하면 물리적 불능인지 게이트
  부작용(보상 차단)인지 구분해 판정할 것 (오라벨이 아니라 커버리지 손실로 나타남)
- balancing 플래너 감속 가정은 brake_ratio(0.25 x lin_accel)로 별도 하향 — 역진자는 뒤로 젖혀야
  감속하는 비최소위상 + LQR 균형 우선이라 유효 감속이 일반의 절반 이하 (일반 가정 시 과속 진입 ->
  관통 루프백 0.8m 실측, 보정 후 루프 소멸·도달 22.3s -> 13.9s)
- omni holonomic의 정상상태 횡오프셋(기준선 아래 약 0.2m 평행 주행, 도달 반경 내) 미해소 —
  평지 무해, 3단계 지형에서 반경 마진 잠식 가능성만 기록 (P 추종 한계, 롤러 접촉 편향 추정)
- balancing은 스폰 시 바퀴 게인 0 override 필수 (02 진입점 패턴 참조)
- WVN식 안정성 점수는 JointTargets.cmd 대비 실측 속도로 계산하고, 개체별 평지 기준선으로
  정규화해 지형 성분만 남길 것. 셀 통과 판정은 반경 크기 비례 + 체류(또는 속도 상한) 조건
- 평지 전제 상수의 지형 한계: tip_accel은 평지 준정적 유도, balancing pitch_eq는 평지 평형,
  PP 기하는 평면 2D — 3단계에서 전도-실패를 별도 태깅해 사후 분석 가능하게
- mecanum 형평성: 시뮬에서 skid처럼 거동(감수)하는데 URDF에는 45도 롤러 구조 단서가 남음 —
  4단계 파서/보조 손실에서 mecanum 요약 피처 규칙 등 정합 조치 검토 필요 (제로샷 주장 방어)
- 지면·바퀴에는 물리 재질이 없어 PhysX 기본 마찰로 동작 중 — 3단계 지형은 TerrainImporter가
  마찰을 명시하므로 정합. 평지 스폰 씬에서 마찰을 통제하려면 지면 재질 명시 필요
- 튜닝 상수는 controller.yaml로 동결 — 개체 1대 구제용 조정 금지, 여러 개체 재현 원인이 있을 때만 변경

## 2단계: legged 제어기 (scripts/sim/controller/legged/) — pure pursuit 명령 + RL 정책 캐스케이드

구조는 wheeled와 동일: 상위 = core/high_pp(공용 명령 계층), 하위 = RL 정책
(속도 명령 추종 -> 관절 위치 목표). form(quad/hex/humanoid)당 morphology-conditioned
정책 1개 (GenLoco식 — 형태 벡터를 관측에 포함, 관절 수 변동은 슬롯 패딩+마스크).
학습 인프라는 legged/train/에 격리 — 배포 경로(controller)는 train/을 임포트하지 않고,
접점은 low_rl의 입출력 규약과 PolicyBundle(학습 export / 배포 load)뿐.

학습 세팅 = 표준 rough-terrain 보행 RL (legged_gym / Isaac Lab velocity rough 계보):
커리큘럼 지형 위 스폰 + 높이 스캔(RayCaster measured heights) + 발 접촉 체공 보상 +
베이스 접촉 종료 + rsl-rl PPO. 표준에서 바꾼 것 2개 — 토크 페널티를 토크 한계
정규화 비율로 (질량 1-265kg 혼합 학습의 규모 불변성), 명령 상한을 로봇별 Froude
규칙으로 (기립고 비례). 난이도 커리큘럼은 한 학습 안에서 자동: 에피소드 종료 시
"반 칸 이상 걸으면 승급 / 명령 거리 절반 미달이면 강등"을 TerrainImporter
update_env_origins가 원점 재배정으로 수행 (사람 개입·재학습 없음).

### 파일별 요약

- `legged/low_rl.py` : 정책 입출력 규약 + 산출물 수명 (Isaac 비의존 — 학습·배포 공용 단일 출처)
  - 조인트 슬롯: base_link -> 접촉 링크(발) 경로 위 가동 조인트 = 로코모션 (팔·waist는
    경로 밖이라 자연 제외 -> 기립 홀드). 다리 순서 = 장착 위치 정렬 (전방·좌측 우선),
    슬롯 = 다리 x 다리당 상한 (quad 16 / hex 24 / humanoid 12), 빈 슬롯 마스크 0
  - 형태 벡터: 전역(로그 질량·외형·기립고·다리 수·sprawl) + 다리별(장착 위치) +
    슬롯별(축·오프셋·가동 범위·토크/속도 한계·기립각), config 고정 스케일 정규화
  - 관측 조립 assemble_obs = [선속도, 각속도, 중력 방향, 명령(3), 슬롯 관절각 오차/속도,
    직전 액션, 형태 벡터, 높이 스캔(187)] (obs_dim: quad 430 / hex 542 / humanoid 370)
  - 높이 스캔 정규화 = clip(몸체 z - 기립 스폰 높이 - 지점 z) -> 0 = 기립 높이의 평지.
    이 규약 덕에 평지 씬은 센서 없이 0 벡터가 정확한 관측 (배포 eval이 그렇게 동작.
    3단계 지형 롤아웃은 bundle.json의 scan 격자로 RayCaster를 만들어 ControlObs.height_scan에 채울 것)
  - cmd_limits(v_max = froude x sqrt(g x 기립고) 클램프)·loco_gain_overrides(kp = 1.0 x
    토크 한계, kd = 0.025 kp — 실로봇 관례 비율. 기본 드라이브 규칙 10x는 RL 과강성)
    — 학습 샘플·배포 클램프가 같은 규칙 (분포 정합)
  - PolicyBundle: export = TorchScript trace(정규화 포함) + bundle.json(차원·스케일·규칙),
    load/act = 배포 (rsl-rl 미설치 환경 로드 가능, 평균 액션 = 결정적)
- `legged/controller.py` : LeggedRobotController — high_pp(unicycle, creep 0) + 번들 추론.
  decimation·physics_dt를 번들 메타로 검증, 명령은 학습 범위 클램프, vy = 0 고정.
  nav_limits·reset(env_ids)·JointTargets(cmd) 계약은 wheeled와 동일 (3단계 공용)
- `legged/train/train_env.py` : LeggedTrainEnv — rsl-rl VecEnv 구현. SimEnvironment 재사용
  (DirectRLEnv 안 씀), 로봇 K종 x env M = N envs (지형 원점 위 스폰), 그룹별 계산 ->
  전역 패딩 텐서 산포. 그룹별 센서 = RayCaster(높이 스캔, 지형 전용) + ContactSensor
  3종 = 발(track_air_time)·베이스(종료)·다리 중간 링크(무릎 보행 페널티). 접촉 센서는
  물리 스텝마다, 스캔은 제어 스텝마다 갱신 (체공 정밀도 vs 레이캐스트 비용).
  humanoid 분기 = biped 체공 보상 + 발 미끄럼 (발 body 인덱스 find_bodies 조회).
  종료 = 베이스 접촉력 > 1N(표준) / 기울기 70도(보강)
  / 시간 초과(time_outs 분리 — 부트스트랩 규약). 커리큘럼 판정은 리셋 시 원점 이동
  전 위치 기준, 리셋 시 발 접촉 이력 reset (이전 에피소드 가짜 체공 보상 방지).
  terrain_cfg=None = 평지 폴백 (스캔 0, 커리큘럼 없음 — 디버그용)
- `legged/train/rewards.py` : velocity env 표준 이식 + 보행 형태 안전장치 (과거
  실패 사례 2건의 표준 처방):
  - 무릎 보행 차단: undesired_contacts — 다리 중간 링크(고관절-발 경로에서 발 제외,
    low_rl.undesired_links) 접촉 페널티 (legged_gym·ANYmal THIGH 페널티 계열)
  - 2족 점프 차단: humanoid는 feet_air_time 대신 feet_air_time_biped (Isaac H1/G1
    positive-biped — 정확히 한 발 지지일 때만 지속 시간 보상 + 문턱 클램프라 두 발
    동시 점프/끌기가 보상 0) + feet_slide (접지 발 수평 속도 페널티 — 스케이팅 억제)
  - 4족·6족은 legged_gym 표준 feet_air_time (접지 순간 체공-문턱 합산) 유지
  - 토크 페널티만 (tau/한계)^2 정규화로 변경 — 질량 1-265kg 분포에서 절대 토크
    L2는 대형 로봇만 벌점 (형태 간 불균형)
- `legged/train/ppo.py` : train_form — rsl-rl OnPolicyRunner 청크 학습 (청크마다 통계
  로그·곡선 기록·체크포인트), 종료 시 PolicyBundle export
- `configs/rl.yaml` : env(지형 모드·주기·액션·리셋·종료)·scan·gains·cmd·obs·morph·rewards·
  ppo·train 규모 (지형 격자: 파일럿 2x2, 본 학습 10x20 = RL 원본 규모, 종류·치수는
  sim.yaml terrain 공유). obs/morph/cmd/gains/scan/action·decimation은 bundle에 저장되는
  규약 — 학습 후 단독 변경 금지
- 기존 확장: core/base.py (ControlObs에 joint_pos·joint_vel·gravity_b·height_scan 추가,
  make_controller legged 분기 + policy_dir 인자, extract_ctrl_params legged 허용),
  environment.py (spawn_robot_groups·robots·group_slice·step_multi·terrain 프로퍼티 —
  이종 DoF는 그룹별 Articulation, 그룹별 전용 프림명 Robot_gXXX 정규식 분리, 배치 =
  base Cloner 명시 positions: 평지 격자 또는 terrain.env_origins 별칭(커리큘럼 제자리
  갱신이 스폰 원점에 즉시 반영), add_terrain(num_envs)로 임포터가 env 원점 배정)

### 설정·실행 (sim 컨테이너)

- 학습: `/isaac-sim/python.sh tools/02_controller.py train --form quad|hex|humanoid --mode pilot|full`
  - pilot -> check/02_controller/legged/{form}/ (곡선 플롯 + 배포 검증 롤아웃 플롯·렌더)
  - full -> /data/EA-Trav/sim/policies/{form}/ (plan 폴더 구조)
  - 학습 셋 = 해당 mode usd 루트 {form} 이름순 앞 num_robots종 (01_sim 변환 선행 전제)
- 평가: `... 02_controller.py eval --mode pilot|full [--robot FORM/NAME]` — wheeled 전체 +
  정책 번들이 있는 legged form만 포함 (없으면 건너뛰고 경고). legged full eval = 동적 검사

### 스모크 테스트 결과 (본 학습 아님 — 코드 실행 검증만, 사용자 지시로 학습 보류)

- Isaac 비의존부 (train 컨테이너): 파일럿 legged 9대 슬롯 배정 전수 정상 (quad 2/3절
  혼재 12-16 active, humanoid 고관절 축 순열 대응, hex 24/24), 번들 export/load 왕복·
  관측 조립 차원·게인/명령 규칙 확인
- sim 컨테이너 파일럿 train (지형 2x2 커리큘럼, form당 2종 x env 8, 10 iter, 각 22초):
  quad(표준 체공 보상 경로)·humanoid(biped 보상 + 발 미끄럼 경로) 모두
  지형 스폰 -> 센서(스캔 187 + 접촉 3종) -> 학습 루프 -> export ->
  make_controller 로드 -> 시나리오 롤아웃 -> 플롯·렌더 전 경로 예외 0.
  미학습 정책의 즉시 전도(에피소드 0.1-2.5s)는 예상 거동
- 종료 판정 오탐 진단 (제로 액션 기립 홀드 150스텝 x 8env): 기울기 판정 0회,
  베이스 접촉 판정 1200회 중 11회 (계단 칸 스폰 저상 개체의 실제 몸통 접촉) —
  종료 로직 정상, 학습 초기의 짧은 에피소드는 정책 미숙 (legged_gym 초기와 동일)
- wheeled 회귀: diff_0000 eval 단건 3/3 도달 12.93s (high_pp 이동·진입점 재작성 후 기존 거동 유지)

### rsl-rl 3.0.1·Isaac 센서 실측 특성 (코드에 반영)

- learn()의 store_code_state가 log_dir None을 처리하지 못함 (무가드 os.path.join) ->
  log_dir 필수 (policies/{form}/rsl_log — tensorboard 이벤트도 여기 쌓임)
- VecEnv 계약: get_observations() -> TensorDict("policy" 그룹), step -> (obs, rew, dones, extras),
  time_outs는 extras로 분리. 정책·정규화는 policy cfg(actor_obs_normalization)로 활성
- export는 actor + actor_obs_normalizer를 deepcopy 후 cpu trace (runner 상태 불변)
- RayCaster: attach_yaw_only는 폐기 예고 — ray_alignment="yaw" 사용 (업데이트마다
  경고를 뿜어 로그가 묻히는 것 실측). mesh_prim_paths=/World/ground는 지형 임포터
  경로와 동일 (RL 원본 정합 — 01 단계에서 예고했던 구성)
- ContactSensor는 스폰 UsdFileCfg activate_contact_sensors=True 전제 (없으면 생성 실패).
  발 센서는 physics 스텝마다 update해야 체공 시간이 정확하다
- print()는 close_app(fastShutdown)에서 버퍼가 유실된다 — 진입점·스크립트는 logger
  또는 flush=True 사용 (01 단계 "close 전 logger" 규약의 stdout 버전)

### 특이사항 (나중에 고려)

- 3단계 지형 롤아웃의 스캔 배관: bundle.json scan 격자로 로봇당 RayCaster를 만들어
  ControlObs.height_scan에 채울 것 (평지 eval은 현재처럼 None -> 0 벡터가 정확)
- 본 학습(full) 전 확인: K=24 그룹 + 센서 4종/그룹의 스텝 처리량 (파일럿은 2그룹만
  실측), PD 게인(kp = 1.0 x effort)·보상 가중치의 form별 적정성, humanoid는 반복 수
  여유 필요 (파일럿 홀드에서도 전도하는 form — 01 단계 기록)
- 보행 형태(걸음새) 품질은 스모크로 검증 불가 — 본 학습 승인 시 첫 단계로 로봇
  1-2종 단기 학습(수백 iter) 후 렌더로 걸음새 확인(점프·무릎 보행 재발 여부)을
  거치고 나서 form 전체 본 학습에 들어갈 것 (보상 가중치 조기 보정 기회)
- 스캔 격자(1.6 x 1.0m)는 전 로봇 공통 (ANYmal 관례) — 전장 0.14-1.9m 분포에서
  소형 로봇에게 과대, 대형에게 빠듯할 수 있음. full 학습 성능 보고 로봇 크기 비례
  격자 검토 (bundle 규약 변경 사항)
- 지형 칸 스폰이라 계단 위 저상 개체는 스폰 직후 베이스 접촉 종료가 드물게 발생
  (진단 실측 1200회 중 11회) — 커리큘럼 강등으로 자연 완화되나, 잦으면 스폰 자세
  보정(칸 중앙 플랫폼 우선) 검토
- 다리 순서는 장착 위치 정렬이라 비대칭 지터 개체에서 좌/우 슬롯 순서가 뒤바뀔 수
  있음 (전후 지터가 좌우 우선순위를 이김) — 형태 벡터에 장착 위치가 있어 정책이
  구분 가능하다는 전제. full 학습에서 문제 시 정렬 키 재검토
- 표기 랜덤화(4단계 입력측 증강)는 이 단계 무관 — 학습·배포 모두 정본 URDF 슬롯 사용
- 매 learn() 호출(청크)마다 rsl_log에 model_{iter}.pt가 쌓임 (rsl-rl 종료 저장) —
  본 학습 후 정리 대상 (배포 산출물은 policy.pt + bundle.json + checkpoint.pt)
- 지형 시드는 여전히 미고정 (TerrainGenerator 전역 난수 — 01 단계 기록과 동일):
  학습에는 무해(분포만 중요), 3단계 GT 생성 때 시드 체계 편입 필요

### 파일럿 학습 결과 (걸음새 확인용 단기 학습 — quad 3회, humanoid 3회 시도)

파일럿 규모: form당 로봇 2종 x env 512, 600 iter, 지형 4x4 커리큘럼 (약 55분/런).
로그: check/02_controller/{quad,humanoid}_pilot_train.log + 정책·곡선·프레임은
check/02_controller/legged/.

- quad: 3차 시도에서 안정 학습 성공 (수익 0.2 -> 7.2, 에피소드 12-14s, 붕괴 없음)
  - 1차 실패 = 자멸 정책 (스텝 보상 순음수 -> 조기 종료가 이득. 100 iter에 붕괴)
    -> 총보상 하한 0 클립 (only_positive_rewards)
  - 2차 실패 = 보상 불균형 붕괴 (페널티 합 -0.9 > 추적 +0.77 -> 클립 0 고착 지대에서
    정책 표류, iter 265-300 붕괴) -> action_rate·dof_acc 절반 + init_noise_std 0.8
  - 4차 (종료 감점 도입 후): -4.0은 250-300 붕괴가 회복 불능 (2회 재현) -> -1.0으로
    역대 최고 안정 학습 (최종 수익 9.47, 에피소드 17.1s, 전 구간 무붕괴 —
    재샘플 10s·리셋 유예 포함 설정. quad_0000이 배포에서 67s 무전도 생존으로
    개선, 단 목적지 추종은 여전히 미달 = 본 학습 반복 수 필요)
  - 걸음새 판정: 주행 중 프레임에서 발끝 접지 정상 스탠스 — 무릎 보행·점프 없음
    (안전장치 작동 확인). 단 "안정 기립" 국소해 — 전진 미학습 (아래 발견 3건)
  - 개체 불균형: 저상 quad_0001 기립 안정 (57s 무전도), 장신 quad_0000 즉시 전도
    — 로봇 2종 학습의 한계, 본 학습(24종) 형태 일반화가 관건
- humanoid: 이 규모에서 부트스트랩 실패 (에피소드 0.9s 정체 — 종료 감점 추가 후에도
  200 iter 무변화). 제로 액션 진단 = 기립 자세는 수 초 유지 (역진자 표류 7s에 60도)
  -> 환경 결함 아님, 능동 균형의 탐색 난이도 (Isaac H1/G1은 4096 env 규모로 해결).
  가설 추가: 생성기 토크 한계가 관대 (350-990 Nm, torque_margin 최대 948)해서
  kp = 1.0 x effort가 질량 대비 과강성 -> 탐색 노이즈 ±0.75 rad 목표가 격렬한
  요동이 되어 0.9s 자가 전복 (quad는 정적 안정이라 생존)

### 파일럿이 만든 설정 변경 (rl.yaml — 전부 실측 근거 주석 포함)

- rewards.only_positive_rewards: true (자멸 정책 차단, legged_gym 표준)
- rewards.termination: -4.0 (클립 뒤 적용 — 능동 균형 form의 전도 비용. Isaac H1 상당)
- rewards.dof_acc -1.25e-7 / action_rate -0.005 (표준의 절반 — 클립 0 고착 방지)
- ppo.init_noise_std 0.8 (노이즈-페널티 되먹임 완화)
- env.resample_cmd_s 5 -> 10 (Isaac 원본 정합 — 5s는 에피소드당 방향 전환 4회로
  순변위가 커리큘럼 승급 문턱 4m을 구조적으로 못 넘는 것 실측)
- cmd.deploy_ratio 0.7 (배포 PP가 학습 상한 속도를 상시 명령 -> 추적 보상 희박
  영역에서 정책 동결 실측 — 중속 대역으로 하향)
- 리셋 후 2 제어 스텝 종료 유예 (정착 충격이 접촉 센서에 수십만 N 스파이크 실측)

### 본 학습 전 필수 과제 (파일럿 결론)

1. **처리량 최적화**: GPU 사용률 9-24%, 약 8-10s/iter (1024 env) — 파이썬 오버헤드
   (그룹 루프·센서 갱신·rsl-rl 외 동기화) 지배. 항별 float 제거·센서 50Hz 갱신으로
   일부 개선했으나 full(3072+ env, 3000 iter)에는 부족. replicate_physics 경로,
   그룹 루프 배치화 검토
2. **humanoid 스케일업**: env 4096급 + (필요시) 게인 규칙 재설계 — kp를 토크 한계
   비례 대신 중력 부하 비례로 (생성기 토크 한계가 관대해 과강성). 파일럿 재검증 후
   본 학습
3. quad 계열 보행 완성도: 600 iter는 기립까지 — 전진 추종은 본 학습 반복 수
   (1500+)에서 창발 예상. termination 감점 추가 후 quad 회귀 확인도 본 학습 전 수행

### 시각 확인(visual check) 개편 — 정지 이미지 -> 주행 비디오 (사용자 지시 + 피드백 반영)

- 산출물: 씬별 궤적 플롯(plots/) + 주행 비디오(videos/{로봇}__{씬}.mp4).
  정지 렌더 저장은 02 평가에서 제거 (01_sim의 스폰 정지 렌더는 유지 — 용도 다름)
- 비디오: H.264(avc1)/yuv420p, 캡처 0.15s 간격 = 6.7fps 실시간 재생, 로봇 추적
  카메라 (외형 치수 비례 거리). VSCode 내장 미리보기 재생 확인 (ffprobe 검증)
- 화면 구성 (피드백 반영 후 확정):
  - 카메라 = 체이스캠 (로봇 헤딩 후방 상공 -> 전방 주시, 시점 저역 필터로
    선회 시 홱 돌기 방지 — 고정 사선 시점은 동작이 안 읽히는 것 피드백)
  - 궤적 = 연속선 (0.4s마다 이전 위치와 잇는 방향 정렬 선분, 지면 높이 부착 —
    몸체 높이 공중선은 떠 있는 전선처럼 보이는 것 실측)
  - 목적지 = 바닥 디스크(반지름 = 도달 판정 반경) + 공중 발광 구 (기둥형은
    장애물로 오독, 반투명 opacity는 RTX 실시간 미렌더 실측). 표면 높이는
    PhysX raycast_closest 샘플 (경사·계단 대응, env.reset 이후 유효)
  - 지형 = 무채색 + 면색: 0.5m 빈 평균 법선으로 분류해 평지 회색 / 경사면은
    4방위별 저채도 색조 / 수직면(계단 챌면) 진회색 (경사마다 바닥색 구분
    피드백). displayColor 면 primvar — 재질 바인딩은 지형 메시에 안 먹고,
    삼각형별 분류는 하이트필드 높이 양자화(계단화 경사) 때문에 체커 얼룩
    실측 -> 빈 평균 필수. 법선은 위쪽 정렬(winding 혼재 대비). 평지 씬 = 격자선
  - 로봇 = 구동부(바퀴·다리 — 이름 토큰 매칭) 검정 2톤 고정 + 몸통 form색
    팔레트 [base, light, 보색 accent] (구동부 검정 피드백)
  - 조명 = 40도 기울인 태양광 2500 + 돔광 150 (수직광은 경사면 음영 없음,
    돔 400은 과노출 실측)
- **UsdShade 바인딩 함정 (2회 실측)**: 스포너 루트 visual_material과 조상 프림
  바인딩은 strongerThanDescendants라 하위 개별 바인딩을 덮거나 메시에 안 닿는다
  -> 로봇은 루트 재질 없이 전 링크 개별 바인딩, 지형은 displayColor primvar
- 시각 프림은 no_shadow(primvars:doNotCastShadows) — 마커 그림자 덩어리 제거
- legged 지형 씬(경사·계단)은 bundle.json scan 격자로 RayCaster를 만들어
  ControlObs.height_scan을 채운다 (3단계 스캔 배관의 선행 구현 — 학습 관측
  정합. quad 계단 비디오로 실전 통과 확인)
- eval 재생성 결과 (wheeled 15대 x 2씬 = 30 비디오): 평지 15/15 통과, 경사
  9/15 — 실패 6건 = 전복 2 (diff_0002·wheeled_humanoid_0001) + 등판력 부족
  정지 4 (diff_0000·skid_0000/0001·omni_0001). 개체별 경사 능력 분포가
  비디오로 판독됨 (시각 확인 목적 달성)
- 보고 키가 "{form}/{이름}::{씬}"으로 바뀜 (full 동적 검사는 평지 1씬이라 영향 없음)

### RL 세팅 독립 검증 (에이전트 3라운드, 합의 성립 — "서있지도 못함" 원인 규명)

검증 방식: 에이전트가 코드 정독 + Isaac/생성기 소스 대조 + 논문 실세팅 조사
(GenLoco·URMA·ManyQuadrupeds·Isaac H1/G1·HumanoidGym·legged_gym 원본) ->
수정 -> 재검증 반복. 핵심 판정 3건:

1. **게인 규칙이 근본 원인** — "kp = 토크 한계 비율 1"은 Go2·ANYmal 표본 2개의
   우연 (토크 한계가 정직한 로봇에서만 성립). 문헌 전반은 **질량 비례**
   (GenLoco kp ≈ 8 x 질량, ManyQuadrupeds 계층표 동일 직선, H1 다리 비율 0.5,
   발목 0.2). 생성 로봇은 토크 한계가 관대(고관절 350-990Nm)해 2-5배 과강성
   -> 탐색 노이즈가 보행 대신 요동·전도 (humanoid 0.9s 자가 전복, quad
   dof_acc 페널티 폭주의 공통 뿌리) -> **질량 비례 규칙으로 교체**
   (kp = kp_per_kg x 질량 클램프, humanoid 발목 kp·kd 별도)
2. **클립 0 고착 = 기립 국소해의 기전** — 비영 명령에서 서 있으면 추적 보상
   ≈ 0, 페널티가 이를 넘어 총합이 하한 클립에 걸림 -> 기립 지점에서 보상
   지형이 평평 -> 전진 기울기 0. **URMA식 페널티 램프**(300 iter 선형)로
   초기 탐색 구간의 클립 발동을 차단, dof_acc·action_rate는 표준값 복원
   (반값 완화는 증상 완화였음). 종료 감점은 램프 제외 (전도 비용은 처음부터
   온전해야 기립 기울기가 섬 — 라운드 2-3 합의)
3. **지형 배정 버그** — Isaac 임포터의 지형 종류 열은 env 연속 블록 배정인데
   로봇 그룹도 연속 구간이라 로봇 종 x 지형 종이 교락 (full이면 로봇 1종 =
   지형 1종). **스트라이드 순열**(전역 g*M+k <-> 임포터 k*K+g)로 해소,
   커리큘럼 갱신·스폰 원점은 순열 경유 (_origins_of가 임포터 직접 읽기)

수정 전체 목록 (라운드 1-2 지적 -> 반영):
- form별 설정 오버레이 (ppo._merge_form_cfg — rl.yaml forms.humanoid):
  termination -4 / orientation -1.0 / lin_vel_z 0 (2족은 수직 진동 필수) /
  joint_deviation(고관절 yaw/roll 이탈 L1, 이름 매칭 + 0슬롯 경고) /
  dof_pos_limits(soft 0.9) / 명령 전진 전용·froude 0.35·v_cap 1.0 —
  전부 Isaac H1/G1 실측값 근거. 병합본이 bundle 저장 -> 배포 재현 일관
- 유예-감점 순서 버그: 종료 판정 -> 유예(자체 _since_reset 카운터 —
  init_at_random_ep_len이 episode_length_buf를 랜덤 대입해 무력화되는 것
  지적) -> 보상 순서로 재배열
- 빈 슬롯 액션 마스킹 (학습 step 초입 + 배포 controller 동일 — prev_action
  관측 노이즈 유입·action_rate 허위 벌점 차단)
- forms.humanoid.cmd에 넣었던 back/vy_ratio 죽은 키 -> env 섹션 이동 (라운드 2)
- 발목 판별은 s % spl >= 4 유지 ("끝 2개" 대안은 2절 다리에서 무릎 저강성
  오판 위험 — 실패 모드 비대칭으로 현행 합의)

보류 합의 (근거 기록): 접촉 이력 종료(처리량 작업과 병행) / deploy_ratio 0.7
(파일럿 성공 후 명령 구간별 추적 오차 곡선으로 재산정 조건부) / obs 노이즈·
DR·푸시(보행 창발 후 강건화 단계) / _global_step 램프는 resume 시 리셋됨
(resume 기능 자체가 없어 특이사항으로만 기록)

파일럿 재실행 관찰 포인트 (검증자 지정): iter 300(램프 완성) 전후 수익 곡선
재붕괴 여부 / 걸음새 판정은 iter 400 이후 렌더로 / 지형 레벨 0 이륙 (국소해
탈출 지표) / humanoid 에피소드 0.9s 돌파 / 빈 슬롯 entropy 팽창은 무시

### 검증 라운드 4-5: 명령 속도 커리큘럼 (quad "기립 고착" 잔여 원인 해소, 합의 4/4)

수정본 quad 재실행 실측: 안정성 전부 해결 (램프 완성 시점 재붕괴 없음, 수익 10-12
고원, 조기 종료 40-100/청크, 장신 quad_0000 배포 67s 무전도·걸음새 정상) —
그러나 지형 레벨 0 고착·배포 전진 0.6m (기립 고착 잔존).

원인 (라운드 4 진단): 추적 시그마 0.25에서 능력 밖 고속 명령(최대 1.9m/s)은
보상 ≈ exp(-14) = 기울기 0인 보상 사막인데, 명령을 상한까지 균등 샘플해서
경험 대부분이 학습 신호 없음. legged_gym command curriculum 누락이 원인.

구현 (원본 GitHub 소스 대조 완료): env별 명령 상한 v_lim = 0.5 시작,
"timeout(완주) 에피소드 + 유효 명령 스텝(‖cmd‖>deadband) 평균 추적 > 0.8"이면
+0.25 확장 (로봇별 v_max 클램프, 축소 없음 = 원본 규약). 검증 지적 반영 2건:
- 정지 명령 스텝은 채점 제외 (기립만으로 추적 1.0 -> per-env 래칫 무임 확장)
- 완주 요구 (생존 스텝 평균은 "질주 후 전도"에 만점 — 원본 고정 분모와 등가,
  약간 더 엄격한 쪽이 래칫 설계에 안전)
부수: 미세 수평 명령 < 0.2 제로화 (원본 규약), 배포 상한 = min(deploy_ratio,
번들 train.cmd_reached_ratio) 클램프 + 경고 (커리큘럼 미도달 번들 보호).
로그·번들에 cmd_v_ratio/cmd_reached_ratio 기록.

관찰 포인트 (검증자 지정): cmd_v_ratio 계단 상승 (시작 0.26) / track_lin 0.38 ->
0.6-0.8 진입 / 확장 직후 일시 수익 하락은 정상 (붕괴 오판 금지) / "cmd_v_ratio
상승 + track_lin 정체 + 레벨 0"이면 무임 확장 시그니처 / 지형 레벨 이륙은
cmd_v_ratio 0.4-0.5 이후 기대 / 유효 명령 0.2-0.23 대역 완주는 기립만으로 통과
(확률 ~1%, 자기 제한 — 초반 계단 노이즈로만 인지)

humanoid 실측 (게인·종료 수정본): 4런 내내 0.9s이던 에피소드가 1.3s로 첫 돌파
(조기 종료 27k -> 19k) — 진행 중

### 검증 라운드 6: 배포 갭 원인 확정 (학습에선 걷는데 배포에선 전도/정지)

명령 커리큘럼 포함 재학습은 학습 지표 전부 개선 (track 0.38->0.70, 레벨 첫
이륙, 붕괴 없음·자가 회복). 그러나 배포에서 quad_0000 즉시 전도(0.26s)·
quad_0001 완벽 기립 후 정지. 판별 진단 연쇄 (전부 실측):

- 기각: 노이즈-진동 보행 가설 (노이즈 주입해도 0.13m/10s), export 버그
  (번들 vs 체크포인트 비트 일치), 명령 OOD 단독 가설 (로봇별 클램프 0.7,
  cmd=0 고정까지 내려도 동일 전도 — 단 검증자 반박: 학습 명령 분포가
  mean 0.11/std 0.21로 좁아져 0.7도 z=2.8 꼬리인 건 사실, 배포 상한 규약
  자체는 교정 대상)
- **확정 1 (전도): 평가 정착(PD 홀드) 상태가 학습 분포 밖** — settle 0s
  (학습 리셋과 동일 시작)면 20s 생존, 0.5s/2s면 0.26s 전도. 이전 정책이
  견딘 건 우연 (전 범위 명령 학습으로 정규화 분산이 컸음)
- **확정 2 (정지): 명령 커리큘럼 동결** — 600 iter에 확장 1-2회. 라운드 4
  채점 강화(유효 스텝 + 완주)가 원본(정지 스텝을 분자에 포함)보다 실질
  과엄해 rough에서 문턱 0.8 미달 지속. 명령 분포가 0.5 부근에 갇힘
- 부수 확인: stairs 씬 "생존"은 역피라미드 피트에 낌 (순변위 판정 필요),
  0001은 커리큘럼 시작값 고착 = 보행 미획득 개체

수정 4건 (라운드 6 합의):
1. legged는 평가 정착 생략 (즉시 정책 제어 — 학습 시작 조건 일치.
   기존 번들로 검증: 0000 flat 0.26s 전도 -> 95s 무전도)
2. 학습 리셋 루트 속도 U(±0.5) 랜덤화 (legged_gym 원본 규약 누락분 —
   전이 상태 내성. rl.yaml env.reset_vel_noise)
3. 배포 상한 = deploy_ratio x 로봇별 도달 상한 (min 아닌 곱 — 전역 평균
   클램프는 v_max 큰 로봇에 z 3.6 명령. train.reached_v를 번들에 기록)
4. curriculum_track_threshold 0.8 -> 0.7 (동결 해소)

판정 규약 갱신 (검증자 지정): 배포는 생존 시간이 아니라 **flat 순변위**로
판정 (낌 오판 방지) / cmd_v_ratio 계단 상승 + 로봇별 reached 시작값 탈출 /
배포 시작 관측 z 전 세그먼트 ≤ 2

### 보행 창발 사다리 재구성 (실험 설계 교정 — 사용자 지적 반영)

로봇 2종 + 거친 지형 + 생성 로봇의 최고 난도 조합에서 전 변수를 동시에
디버깅하던 방식을 폐기하고, 변수 1개씩 추가하는 사다리로 전환:
- 단계 0: 표준형 quad 1대(quad_0001 — 23kg·기립 0.28m·2절) x 평지 =
  파이프라인 보행 존재 증명. rl.yaml train.pilot이 현재 단계를 표현
  (robot_names 필터·terrain_mode 추가)
- 단계 1: rough 지형 -> 단계 2: 2종 공유 -> 단계 3: 본 규모
- 단계 0은 iter당 1.5s (600 iter = 15분) — 반복 루프가 8배 빨라짐

### 단계 0 진단 연쇄 (배포 정지의 최종 원인 수렴)

- 학습 track_lin 0.8대 도달 (게인 2x질량 + 푸시 + 채점 하한 후) — 그러나
  배포 평균 정책은 순변위 ~0.1m/15s 정지
- **3원 판별 (결정적)**: 배포 평균 0.08m / 배포 확률(std 0.81) 1.43m /
  학습env 평균 0.09m -> 환경 갭 아님. **노이즈 디더링이 보행을 대행하고
  평균 정책은 정지하는 구조** — std가 전 런에서 0.8에 고정된 것이 원인
- entropy 0.008 -> 0.002: std 0.80 -> 0.74 (2000 iter) — 감쇠가 구조적으로
  느림 (노이즈가 수익에 기여해 축소 압력 상쇄)
- 학습량 2000 iter (49M 표본 — 표준 100M의 절반): 명령 커리큘럼 0.52 ->
  0.67 재확장 등 진전은 있으나 배포 평균은 여전히 정지
- 조치 (진행 중): **std 상한 선형 어닐링** (init 0.8 -> 0.2, 청크 경계
  클램프 — ppo.py, rl.yaml ppo.std_anneal_to). 탐색 어닐링 표준 기법으로
  평균 정책이 보행을 인수하게 강제

### 제어기 구조 제안 (플랜 변경 — 사용자 승인 대기)

form당 공유 정책(plan 원안) 대신 **GT 경로는 로봇당 정책**으로 전환 제안:
- 근거: 연구의 제로샷 주장은 URDF 인코더(모델) 소관 — 제어기는 롤아웃
  도구라 로봇당이어도 전제 무손상, GT 품질은 오히려 향상
- 비용: 로봇당 15-30분 (단계 0 실측) x 24종 ≈ 반나절 — 병렬화 가능
- 공유 정책(GenLoco식)은 병행 연구 항목으로 강등
- 주의: 로봇당 전환은 현 병목(1대를 결정적으로 걷게 만들기)의 해법이
  아님 — 단계 0이 이미 로봇당 조건. 단계 0 해결이 선결

### 단계 0 반복 실험 기록 (한 놉씩 — legged_gym 평지 완전 정렬로 수렴)

각 런 2000 iter x 1024 env (약 50분), 변경 1건씩. 판정 지표 = walked_m(순변위,
금본위)·track_lin(정직화 후)·feet_air 부호·배포 flat 순변위:

- entropy 0.008 -> 0.002: std 0.80 -> 0.74 감쇠 시작 (구조적 저속 — 노이즈가
  수익 기여해 축소 압력 상쇄). 배포 정지 지속
- std 상한 어닐링 (0.8 -> 0.2): **상한 0.44 부근 정책 전면 붕괴** (에피소드
  0.9s, 전도 33k) — 노이즈 의존 보행이 평균으로 이식되지 않고 소멸. 어닐링은
  보행이 강해진 뒤 완만 목표로만 재시도 (config std_anneal_to: null)
- feet_air_time 0.125 -> 1.0 (legged_gym 원본): 단독으론 체공 음수 지속
- **체공 문턱 크기 비례화** (고정 0.5s -> 2.1 x sqrt(기립고/g), [0.2, 0.6]):
  소형 로봇(기립 0.28m -> 자연 체공 ~0.3s)은 정상 걸음까지 전부 벌점이던
  크기 비스케일 함정 — 게인과 동일 부류. ANYmal 대입 0.49로 원본 정합
- **추적 보상 속도의 정직화**: 관측 순간 속도 -> 제어 주기 평균(변위/시간,
  몸체 프레임 회전). "track 0.83 + 순변위 0.4m" 모순 조사 중 도입 — 결과
  모순의 주원인은 진동 해킹이 아니라 아래 명령 분포였음이 판명됐지만,
  해킹 가능성 차단 차원에서 유지 (관측은 순간 속도 그대로 — 배포 규약 불변)
- **명령 분포 압력 교정** (v_start 0.5 -> 1.0): 시작 0.5는 평균 |명령| 0.25
  -> 정지 track 0.78인 무압력 분포 (걷기의 한계 이득 소멸 — 기립 국소해의
  최종 기전). legged_gym이 처음부터 ±1.0을 쓰는 이유
- **quad 종료 감점 제거** (-1 -> 0, humanoid -4 유지): legged_gym 4족은 종료
  감점이 없음 — 감점은 "걷기 시도 = 초기 잦은 전도"의 비용이 되어 기립
  고수를 강화. 자멸 유인은 only_positive가 차단

현재 런 = 위 전부 적용 (legged_gym 평지와 남은 차이: env 1024 vs 4096,
관측의 상수 패딩 370차원(형태 벡터+제로 스캔), 로봇 자체). 결과 대기

### 단계 0 세션 결산 (미통과 — 병리 정밀 규명 상태로 종료)

legged_gym 평지 완전 정렬 런(전 수정 적용) 결과: 학습은 정직 지표(평균
속도 track) 0.74로 **확률 정책의 실보행 창발** (정지 기준선 ~0.44 대비 명확),
그러나 배포 평균 정책은 순변위 0.004m 완전 정지 (기울기 2.2도 완벽 기립).

병리의 최종 성격: **PPO 평균이 보행을 인수하지 않는 노이즈 래칫 보행** —
탐색 노이즈(std ~0.7, 감쇠 안 됨)가 접촉 비선형을 통해 정류되어 전진을
만들고, 평균 정책은 그 요동의 중심만 잡는 형태로 수렴. 어닐링으로 노이즈를
빼앗으면 보행이 이식되지 않고 소멸 (붕괴 실측).

남은 미검증 차이 (다음 실험 큐 — 판별력 순):
1. **go2 교차 검증**: 실로봇 go2 URDF를 이 파이프라인으로 학습 — 걸으면
   "생성 로봇의 물리(발 형상·마찰·관성)" 문제, 안 걸으면 파이프라인 잔여
   결함으로 이분. 실로봇 풀 meta가 생성기 포맷(standing_pose 등)과 달라
   어댑터 필요
2. **발 마찰 명시**: 로봇 링크에 물리 재질이 없어 발-지면 유효 마찰
   0.5(PhysX 기본 x multiply) — legged_gym DR 하한과 같은 값이라 단독
   결정타는 아니나 교정 대상 (wheeled 캐스터와 동일 부류의 기록된 결함)
3. env 4096 스케일 (처리량 최적화 선행) + 관측 상수 패딩 370차원(형태
   벡터 + 제로 스캔) 제거 실험 (학습 효율 영향 분리)

### 독립 분석: 보행 실패의 원인은 생성 로봇의 물리 (URDF 생성기 재검토 필요)

quad_0001 URDF 직접 정량 분석 + GenLoco/GenBot-1K/X-Nav 대조 결과 (분석 전용
에이전트 — 수정 미적용):

- 확증 1: 현 파일럿 = 로봇 1대 x 정책 1개 (rl.yaml pilot). 단 관측에 상수
  370차원 잔존 (형태 벡터 183 + 제로 스캔 187 + 패딩 슬롯)
- **[치명] 트롯 물리적 배제**: 전면 무릎 토크 한계 13.4Nm < 대각 2지지 정적
  요구 14.06Nm — 가장 배우기 쉬운 보행이 불가능한 개체. 원인 = 생성기 토크
  하한이 스윙 자중만 기준 (multileg.py tau_ref x0.8 하한), validate.py 토크
  검사도 지지(스탠스) 하중 미검사
- **[치명] 좌우 비대칭**: 관절별 독립 랜덤화로 토크 3.0배·속도 2.1배·부착점
  30mm 좌우 편차 — 대칭 걸음 해가 액션 공간에 없음. 토크 포화(노이즈 1σ만으로
  전면 무릎 예산 59%)의 비대칭 정류가 "노이즈는 걷고 평균은 서는" 병리의
  직접 기전
- [높음] 관절 속도 한계 U(5,15)가 크기·보행 요구 비연동 — 전속 스윙 피크
  9.2-9.9 rad/s 대비 관절 3개(7.07-7.96) 미달
- [중간] 균일 밀도 -> 질량 분포 역전 (다리 9.6% — 실로봇 ~50%, 원위 58% —
  실로봇 ~10%), 발 마찰 재질 무기록 (유효 0.5), 정사각·광폭 몸통 허용
  (폭 0.712 > 장 0.460 — 기립 국소해 심화)
- 기각: 다리 스윙 관성 과대 가설 (Go2급과 유사 규모)
- 레퍼런스 결정 차이: 3편 모두 (i) 실로봇 실측값 앵커 근방 랜덤화, (ii)
  액추에이터 한계를 실모터/질량 비례로 로봇 단위 결정, (iii) 다리 대칭
  유지. 우리는 앵커 없는 임의 범위를 관절 단위 독립 샘플 — 레퍼런스에
  존재하지 않는 개체군 생성
- 판정: "학습 불가능"은 아님 (확률 정책 실보행) — 그러나 결정적 평균 보행에
  정량적으로 불리. go2 교차 검증이 최종 이분 실험으로 유지
- 개선 방향 (미적용): 다리쌍 미러링 / 지지 하중 기준 토크 하한 + validate
  스탠스 토크 검사 / 스윙 연동 속도 하한 / 발 마찰 명시 / 힙 집중 질량
  모델 / 몸통 폭·장 비율 제약

### 최종 이분: 스톡 레퍼런스 A/B (결정적)

- go2_proxy (실 go2 관성/조인트 + 블록 형상) 교차 검증: 우리 파이프라인에서
  동일 병리 재현 (정지 수렴) -> 생성 로봇 탓 아님 확정. 발 마찰 1.0 명시·
  셀프 충돌 OFF·솔버 4/0 (Isaac 보행 자산 표준 정렬)로도 동일
- 자인 2건 추가: go2 커리큘럼 0.93은 확장이 아니라 시작값(1.0/1.08) —
  확장 0회. 확률 정책도 배포 env에선 0.16m (학습 지표의 "보행 증거"는
  결국 전무했음 — 전 런 공통 "정지 수렴")
- **스톡 Isaac-Velocity-Flat-Unitree-Go2-v0, 동일 1024 env**: iter 943에
  track 0.90, 완주 생존, **std 1.0 -> 0.40 자연 감쇠** (우리는 0.74-0.8
  고정) -> **env 규모 아님, 우리 커스텀 env의 미발견 결함 확정**.
  std 고정은 원인이 아니라 결함의 증상
- 다음: 스톡 유효 설정 vs 우리 설정 필드 단위 diff (검증 에이전트) —
  유력 미대조 항목: heading 명령 방식(스톡은 wz를 헤딩 오차에서 유도),
  관측 노이즈 NoiseCfg, 이벤트 랜덤화(재질·질량·초기상태 범위), 보상
  가중 정확값, PPO lr/entropy 정확값

### 스톡 diff 기반 소거 실험 (Run A-E — 검증 에이전트 합의 계획, 밤샘 순차)

레퍼런스 확정치: 스톡 Go2 flat, 동일 1024 env, iter 1499 = track 0.92,
std 0.33, 완주 생존 — 우리 합격선.

diff 전 필드 대조 결과 (에이전트 보고 — 상세는 대조표 참조):
- 1위 후보: obs 정규화 ON(우리)/OFF(스톡) — 표현 비정상성 + 저분산 eps
  증폭 자가강화로 "기립 수렴 + std 고정" 단일 기전 설명
- 2위: 명령 구조 (스톡 = heading P제어 + 정지 2% / 우리 = 독립 wz ±1.5
  10s 홀드 + 정지 ~20%)
- 3위: 관측 노이즈 전무 (스톡 corruption: lin ±0.1, ang ±0.2, grav ±0.05,
  qpos ±0.01, qvel ±1.5 — 정밀 정지 과적합 차단)
- 4위: only_positive 클립 + 페널티 램프 (스톡엔 둘 다 없음 — 클립은 초기
  보행 시도 스텝의 구배를 지움)
- 기타 판명: 스톡 Go2는 push 없음(우리 15s 푸시는 스톡에 없는 것),
  reset 속도 0 (우리 ±0.5는 비스톡), 발 재질 0.8/0.6, 네트 [128]x3,
  avg-vel 보상은 무해·무익 판정 (20ms 창 = 순간과 실질 등가)

실험 결과:
- Run A (정규화 OFF): 실패 — std 0.76 고정, 배포 정지. 단 초반 50-150
  iter에 walked 1.88m 실보행 + 커리큘럼 실확장(도달 1.0) 후 기립 회귀
  — "걷기를 찾지만 가치 지형에서 패배" 패턴 첫 관측
- Run B (heading 명령 + 정지 2%, A 누적): 실패 — std 0.76, walked 0.16
- Run C (관측 노이즈 스톡값, A·B 누적): 진행 중
- 남은 계획: D = only_positive·램프 제거 / E = 관측 48차원·네트 [128]x3
  stock 형상 (실패 시 최후 수단 = 스톡 태스크에 우리 로봇 역이식)

### 역이식 실험 — 결함 최종 격리 (밤샘 소거 종결)

- Run C(관측 노이즈)·D(클립/램프 제거 -> 자멸 재현: 리셋 속도 노이즈·푸시가
  스톡 Go2에 없는 가혹 조건이었음을 확인)·S(완전 스톡 정합 설정)·E(관측
  60차원 + 네트 128x3) 전부 실패 — 매 런 std 0.74-0.76 고정
- 스톡 초기 관찰: 에피소드 총보상이 음수로 깊어지며도 에피소드 성장
  (11 -> 689스텝) = 순음수 초기 보상에서도 자멸하지 않음 (클립 불필요 근거)
- **역이식 (tmp_transplant): 스톡 태스크 + 우리 go2_proxy USD만 교체 =
  iter 400에 track 0.77, std 1.0 -> 0.57 건강 감쇠, 보행 성립**
  (check/02_controller/transplant_train.log)
- 판정: 로봇·USD 변환기·설정값 전부 무죄. **결함 = train_env.py 실행
  의미론 (코드 동작)** — diff 에이전트에 코드 수준 최종 감사 위임
  (의심 축: 게인 실적용 여부 — ImplicitActuatorCfg 이름 매칭 실패 시
  조용한 기본 규칙 폴백 = kp 10x effort 과강성 / 스텝 순서 / prev_action
  raw vs 가공 / 리셋 시 PD 목표 잔존 / 접촉 이력)
- 역이식 완주 (1000 iter): 보상 33.1, 에피소드 985/1000 생존, 추적
  1.40/1.5 (93%), std 0.37 — **판정 확정**. tmp_transplant.py 삭제

### 결함 확정 — 리셋 분포 (역이식 후속 계측)

- 런타임 계측 (tmp_dump_ours/stock, 같은 go2_proxy USD 4 env):
  - 게인 실적용 kp 30.03/kd 0.751 (설계값 정확 — 조용한 폴백 가설 기각),
    슬롯-조인트 배선 정상 (FR_thigh 응답 84% 추종), 기립 처짐은 스톡도
    동급 (0.40 -> 0.27) — **순수 동역학 무결**
  - 스톡은 DCMotor(소프트웨어 PD·토크 포화)라 physx 게인 0으로 찍힘 —
    실효 응답은 우리 implicit PD와 대동소이
- 판별 신호: 실패 런 전부 surrogate loss **양수(+0.014)** 지속, 성공
  역이식은 **음수(-0.01)** — 양수 지속 = 어드밴티지가 노이즈 (액션이
  보상 차이를 만들지 못함)
- 원인: **리셋 분포**. 우리 env는 기립 리셋(관절 ±0.05, 속도 0)이라
  랜덤 정책에서도 조기 종료 0건(첫 이터부터 20s 완주). 스톡은 관절
  0.5-1.5배 스케일 + 루트 6축 속도 ±0.5 + xy ±0.5/yaw ±π + 10-15s
  푸시로 초기 에피소드 14스텝(즉시 전도) — 전도·회복이 학습 신호의
  원천인데 우리 분포에는 그 신호가 0
- 오판 정정: Run D/S의 "스톡 Go2는 리셋 속도 노이즈·푸시 없음"은 잘못된
  판독 — velocity_env_cfg 원본에 둘 다 존재 (해당 근거로 껐던 설정이
  결함을 오히려 강화)
- 수정: rl.yaml env 리셋 키 재편(reset_joint_scale [0.5,1.5],
  reset_pos_xy_noise 0.5, yaw_noise 3.14, reset_vel_noise 0.5, push
  10s/0.5) + train_env._reset_envs를 스케일 리셋(soft 한계 클램프)으로
  재작성. joint_noise 키 제거
- 검증 기준 (파일럿 재학습): 초기 에피소드 급감 + 조기 종료 다수 +
  surrogate 음수 전환 + std 0.8 -> 0.5대 감쇠 + walked_m 상승
- 리셋 분포 단독 수정 판정 (600 iter): 실패 재현 — 보행 2.0 -> 0.59 감소,
  std 0.76 고정, surrogate 양수 복귀. 조기 종료가 스폰 직후(77건)에만 잠깐
  발화하고 정책이 곧 회피 (400 iter부터 종료 급증하는 다른 국면도 관찰)
- 추가 수정: **종료 판정 강화** — (1) 베이스·중간 접촉 이력 3틱 max 판정
  (순간 50Hz 샘플은 충격 접촉 과소 발화, 스톡 history 규약), (2) 기립고
  붕괴 컷 term_height_ratio 0.5 (무릎·배 깔림은 베이스 접촉·기울기 판정을
  모두 회피하는 은신처 — 기립 홀드 처짐 0.71x / 붕괴 0.3-0.45x 실측 사이).
  지면 고도: 지형 = 스캔 중앙값, 평지 = 0
- 좀비 정리: app.close()가 안 죽고 CPU 점유 (transplant·dump 3건 kill -9).
  tmp_dump_* 삭제
- 종료 강화 1차(이력 3틱 + 붕괴 컷) 판정: 무효 — 시드 고정 하 수치가 이전
  런과 완전 동일 (붕괴 컷 발화 0). 원인: kp 30 PD가 접힌 관절을 약 100ms에
  복원해 관절 스케일 스폰만으론 붕괴 자세가 지속되지 않고, 스톡의 초반
  대량 종료의 실체는 **착지 순간 베이스 충격 접촉** (스톡 ep_len 26@100it
  = 진짜 조기 전도. 다만 초반 14스텝은 init_at_random_ep_len 부기 잔상
  섞임 — 오독 주의)
- 종료 강화 2차: (1) 베이스 접촉 센서만 물리 스텝 200Hz 갱신 x 이력 4틱
  max (착지 충격 1-2 물리 스텝을 50Hz 스냅샷이 60-75% 놓침), (2) 종료
  유예 제거 (termination_grace_steps 0 — 유예 3스텝이 스폰 착지 충격
  종료를 정확히 억압. "무너짐 = 즉시 끝"이 가치 신호의 원천이므로 억압
  대상 아님). 나머지 센서(발·중간·스캔)는 50Hz 유지 (처리량)
- 종료 강화 2차 판정: 시드 고정 수치 여전히 동일 = 이 로봇은 스폰 랜덤화
  로는 안 무너진다 (PD 복원 약 0.1s가 낙하 0.2s보다 빠름 — 착지 전에
  다리가 펴져 베이스 충격 자체가 희소). 종료 강화는 유지 (이후 지형·
  다임바디먼트에서 유효)
- **정합 누락 발견**: 스톡 go2 flat PPO는 entropy_coef 0.01 /
  init_noise_std 1.0 — 우리는 0.002 / 0.8 (Run S가 탐색 계수 2종을
  누락). 과거 하향 근거 실측들은 전부 종료가 죽은 구 환경의 증거라 무효
  판정, 스톡 값 복원 (run-4)
- **run-4 (탐색 계수 복원) 국면 전환**: iter 150에서 조기 종료 11,107건
  (에피소드 1.7s) — 대량 전도 국면 진입 후 생존 회복 궤도 (1.7 -> 4.3s,
  수익 -0.18 -> 0.67 상승, 종료 감소). 스톡 성공 시그니처와 동형. std 1.0
  + entropy 0.01의 대담한 탐색이 리셋·종료 정합과 결합해 처음으로 "전도가
  존재하는" 학습 신호를 만들었다. 판정: 결함은 단일 원인이 아니라
  **리셋 분포 + 종료 발화 + 탐색 계수의 3중 결합** — 셋 다 스톡과
  달랐고 각각 단독 수정으로는 국면이 안 바뀌었다

### 스톡 클론 모드 (run-5) — 검증된 역이식 세팅의 전 축 복제

- run-4 판정: 대량 전도 국면은 열렸으나 "7s 생존 + 무보행" 고원 (std 0.97
  고정, surrogate 양수). 다른 세션이 term_height_ratio 0으로 끔 (웅크림
  스폰 즉사 루프 진단 — 회복 경험 차단) -> 수용
- **최종 정정**: 스톡 go2 전용 cfg가 velocity 베이스 클래스의 험한 리셋을
  도로 끈다 (관절 스케일 (1,1)·리셋 속도 전부 0·push None·undesired None·
  base_com None·mass (-1,+3)). Run D/S의 원판독이 옳았고 내 "오판 정정"이
  재오판 — 역이식 성공 세팅의 리셋은 포즈 xy ±0.5/yaw ±π 뿐. code.md의
  "리셋 분포 결함" 서사는 폐기
- 스톡을 넘어뜨리는 실제 원천 = **DCMotor 속도-토크 포화** (계측: 같은
  명령에 스톡 응답 과도 0.39/목표 0.25 — 저제동. 우리 implicit 무포화는
  0.21 수렴 — 과제동으로 전도 희소)
- run-5 변경 (전부 역이식 검증값으로): actuator_model dcmotor (신규 —
  robot_spawn._dcmotor_actuators, 토크한계별 그룹, URDF effort/velocity,
  번들 기록·평가 재현), track_avg_vel false (순간 속도 복귀), 관측 스케일
  전부 1.0 (스톡 무스케일), 리셋 원복 (스케일 (1,1)·속도 0·푸시 끔),
  명령 vx (-1,1)/vy ±1, 기울기 종료 비활성 (베이스 접촉 단일 종료)
- 잔여 미정합 (의도적 보류): 게인 kp30/kd0.75 vs 스톡 25/0.5 (질량 비례
  규칙 유지), add_base_mass DR (-1,+3) 없음, 발 재질 1.0 vs 0.8/0.6
- run-5 판정: **이중 실행 사고** (미상 프로세스와 동시 기록 — 산출물 오염,
  이후 실행 전 프로세스 수 검증 절차 추가) + 클론 config 자체의 궤적:
  초반 유망 (100-250 iter 순보행 3.4-3.5m 역대 최고, 수익 -16 -> -9.5
  회복 중) -> 700-1300 iter 즉사 루프 (에피소드 0.5s) -> 실패. 기전 =
  초기 순수익 -16/에피소드(포화 모터 토크 페널티)에서 종료 무비용이라
  "빨리 죽는 게 이득" 자멸 구배. 스톡이 클립 없이 버티는 건 조기 전도
  창발로 에피소드가 짧아 음수 누적이 작기 때문 (우리 로봇은 안 넘어져
  20s 누적) — Run D의 클립 제거가 오히려 탈정합이었다
- run-6: only_positive_rewards true 복원 (단일 변경). 산출물 삭제 후
  단독 실행 + 프로세스 수 검증
- run-6 (클립 복원) 600 iter: 자멸 루프 차단 성공 — 에피소드 20s 유지,
  순보행 3.5m 지속 (최장 유지). 550부터 보행 하락 조짐 (2.8 -> 1.9) +
  std 1.02 고정·surrogate 양수 잔존. 완주 후 자동 평가(비디오)로 평균
  정책 거동 확정 예정. 다음 런 후보 (잔여 미정합): 게인 kp30/kd0.75 ->
  스톡 25/0.5 정합, 발마찰 1.0 -> 0.8/0.6, add_base_mass DR (-1,+3)

## 2단계 방향 전환 (사용자 지시): 커스텀 RL env 소거 루프 중단 -> 스톡 세팅 채택

- 결정 사항 (사용자 지시 3건):
  1. legged 제어기는 **로봇당 정책 1개** (form당 공유 정책은 폐기 — 이전 지시가
     "승인 대기"로 방치돼 있었음. 제로샷 주장은 URDF 인코더 소관이라 전제 무손상)
  2. RL 설정(학습 env 코드 + 하이퍼파라미터)은 직접 만들지 않는다 —
     **기존 오픈소스 go2 RL 세팅을 그대로 사용**
  3. URDF 문제 발생 시 즉시 원인 분리할 수 있도록 **go2 베이스라인 먼저 확보**

### 스톡 go2 학습 파이프라인 확립 (Isaac Lab 2.3.0 내장 태스크)

- 학습·재생 전부 컨테이너 내 순정 Isaac Lab (/workspace/isaaclab, git diff 없음) +
  공식 go2.usd (클라우드 에셋 — 이 컨테이너에서 다운로드 동작 확인)
- flat 태스크(1024 env x 1500 iter, track 0.92)로 보행 성립을 먼저 검증한 뒤
  rough(지형)로 이행 — flat 산출물은 무누적 원칙에 따라 삭제됨
- 재생 커맨드 (sim 컨테이너):
  `cd /workspace/isaaclab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py
  --task <태스크>-Play-v0 --headless --video --video_length 500 --num_envs 16 --checkpoint <model.pt>`
  (H200 헤드리스에서 --video 정상 동작 — gym RecordVideo 경로는 뷰포트 캡처와 달리
  파일 생성됨. --video 모드는 video_length 도달 시 자연 종료)
- 체크포인트 영속화: /workspace/isaaclab은 컨테이너 레이어(비마운트)라 재생성 시
  유실 -> 학습 완료 즉시 /data/EA-Trav/sim/policies/로 백업할 것
- 주의: 컨테이너 내 프로세스 확인·종료 시 `pgrep/pkill -f`는 docker exec 자기 자신의
  명령줄과 매치한다 (BUSY 오탐·자기 종료) — `[t]rain` 브래킷 패턴 사용.
  호스트 nvidia-smi는 8 GPU 전체가 보임 (우리 할당 0-3 확인은 컨테이너 안에서)
- URDF vs RL 판별 프로토콜 (이후 표준): 생성 URDF 로봇이 이 스톡 세팅(로봇 cfg만
  교체)에서 안 걸으면 로봇 물리 문제, 걸으면 파이프라인 문제 — go2 베이스라인이
  대조군. 커스텀 train_env.py 계열은 GT 경로에서 사용 중단

### 스톡 go2 rough (TerrainGenerator) — 학습 완료 + 보행 검증 완료 (사용자 지시: 학습 환경은 지형)

- `Isaac-Velocity-Rough-Unitree-Go2-v0` 스톡 레시피 그대로 (4096 env x 1500 iter,
  ROUGH_TERRAINS_CFG 커리큘럼 지형 + 높이 스캔 — plan GT 롤아웃 지형과 같은 생성기 계열)
- 결과: 약 1.5시간 완주, 최종 평균 보상 21.9 / 에피소드 950/1000스텝 (중반 커리큘럼
  승급 구간에서 보상 17.6·에피소드 910으로 출렁 후 회복 — 정상 패턴)
- 보행 검증: rough Play 비디오에서 랜덤 러프·박스·계단 지형 가로질러 이동, 전도 0
  (check/02_controller/go2_stock/go2_rough_play.mp4). 비디오 캡처 후 play 프로세스는
  자체 종료함 (flat 때처럼 pkill 불필요했음 — 거동 비일관, 종료 확인은 항상 pgrep으로)
- 체크포인트: /data/EA-Trav/sim/policies/go2_stock/{flat,rough}/model_1499.pt + params/
- 다음 단계 (로봇당 정책 경로): 생성 URDF 로봇을 이 rough 태스크에 로봇 cfg만 교체해
  이식 (역이식 패턴 상속) — go2 대조군과 동일 세팅에서 학습해 URDF 문제를 즉시 분리

### go2 rough 정책 단차·걸음새 진단 (사용자 관찰: "계단이 안 보이고 발을 끈다")

- 계단은 학습 지형에 있음: ROUGH_TERRAINS_CFG = 계단 40% (정·역피라미드 각 0.2,
  단차 0.05-0.23m 난이도 스케일), go2 오버라이드는 박스·러프만 축소하고 계단은 원값.
  play 비디오에 계단이 드문 건 Play cfg가 5x5 축소 격자 + 커리큘럼 off 랜덤 스폰이라
  계단 칸 배정이 적었던 것
- 커리큘럼 도달: Curriculum/terrain_levels 3.5 -> 5.5/9 (1500 iter 종료 시점에도 완만
  상승 중 — iteration 연장 시 개선 여지). 단차 환산 약 0.16m까지 학습 분포에서 소화
- 계단 전용 정량 평가 (Play 태스크 hydra 오버라이드로 계단 100%·난이도 1.0 = 0.23m
  고정, 64 env x 10s, exported jit 정책): 전도 11/64 (17%), 수평 변위 mean 4.26m /
  median 4.0m, 3m 이상 이동 55%, 일부 개체 제자리(min 0.02m). 0.23m 계단 등반
  장면은 비디오로 확인 (해당 판 비디오는 재학습 채택 후 무누적 원칙으로 삭제)
- 걸음새: 터치다운 체공 mean 0.32s / p90 0.41s (트롯 정상 범위 — 스윙 타이밍은 정상).
  "발 끌기"의 실체는 클리어런스(들어올림 높이) 부족 — 원인은 스톡 go2 rough 보상 구조:
  feet_air_time 가중 0.01 (에피소드 기여 -0.001 = 죽은 항) + foot clearance 항 부재.
  참고: go2 flat cfg는 feet_air_time 0.25 (rough만 0.01로 낮춰져 있음)
- 측정 함정 (기록): 50Hz 스냅샷 기반 접촉 듀티 팩터는 과소 측정 (0.15로 나옴 —
  물리적 불능치) — 접촉 통계는 센서 내부 적분값(last_air_time 등)만 신뢰할 것.
  print 버퍼는 PYTHONUNBUFFERED=1 필요 (fastShutdown 유실 — 01 단계 기록의 재확인)
- 하이드라 오버라이드로 지형 조성 변경 가능 확인 (play.py hydra 지원):
  `env.scene.terrain.terrain_generator.sub_terrains.{이름}.proportion=` +
  `...terrain_generator.difficulty_range=[d,d]` — 표적 평가 표준 방법으로 채택
- 뒤로 걷는 관찰의 원인: 스톡 명령 샘플이 lin_vel_x U(-1,1) — 절반이 후진 명령이고
  로봇은 그걸 추종한 것 (학습·렌더 결함 아님. 비디오 초록 화살표 = 명령)

### go2 rough 재학습 (진단 반영) — 현행 표준 세팅

- 스톡 train.py에 하이드라 오버라이드 3건만 (소스 무수정):
  agent.max_iterations 1500 -> 3000 (커리큘럼 미완 해소),
  env.rewards.feet_air_time.weight 0.01 -> 0.25 (flat cfg 값 이식 — 발끌기 교정),
  env.commands.base_velocity.ranges.lin_vel_x [-1,1] -> [-0.5,1] (전진 위주 — 사용자 지시)
- 학습: 3시간 완주, 최종 보상 26.4, 커리큘럼 레벨 5.5 -> 6.1/9, 전도 종료율 7.2% -> 4.0%
- 계단 d1.0(0.23m) 정량 비교 (재학습 전 -> 후, 동일 프로토콜 64 env x 10s):
  전도 11 -> 4 (17% -> 6%), 3m 이상 통과 비율 0.55 -> 0.53, 터치다운 체공 0.32 -> 0.36s
  (변위 mean 4.26 -> 3.70m은 퇴보 아님 — 전진 편향으로 평균 명령 속력 자체가
  0.5 -> 0.42로 줄어 비교 불가 지표. 전도율이 유효 비교축)
- 산출물 (무버전 단일본 — 사용자 규칙: 산출물 누적 금지, 버전 접미사 금지.
  갱신 시 이전 것 삭제 후 같은 이름으로 교체):
  - 정책: /data/EA-Trav/sim/policies/go2_stock/rough/ (model_2999.pt + params +
    exported policy.pt — jit 단독 로드 가능)
  - 비디오: check/02_controller/go2_stock/go2_{fwd,lat,stairs_fwd}.mp4
- visual check 표준 프로토콜 (사용자 지시: 랜덤 명령 대신 고정 명령):
  전방 0.8m/s / 횡 0.5m/s / 계단(0.23m) 전방 0.8m/s 3편. 고정법 = play 오버라이드로
  `ranges.lin_vel_x·lin_vel_y`를 한 값 범위로, `ranges.heading=[0,0]` +
  `env.events.reset_base.params.pose_range.yaw=[0,0]` (전 로봇 +x 정렬),
  `rel_standing_envs=0.0`. 시각 판정: 3편 모두 정렬 이동·전도 없음 확인
- 잔여 관찰: 계단 최고 난이도 제자리 개체 소수 잔존 (min 0.02m) — 필요 시
  iteration 추가 연장으로 개선 여지 (레벨 6.1도 계속 상승 중이었음)
- 다음 단계: 생성 URDF 로봇을 같은 태스크에 로봇 cfg만 교체해 로봇당 정책 학습

## URDF 생성기 결함 3건 수정 (독립 분석 확인분 — 사용자 지시로 반영)

수정 파일: scripts/urdf/platform/multileg.py (전면 재작성), humanoid.py(_set_limits),
utils/validate.py(스탠스 검사 추가), configs/urdf.yaml(gait 섹션 신설)

1. **스탠스 토크 하한**: 종전 하한(스윙 자중 스케일 tau_ref x 0.8 x 여유 0.8)은
   트롯 지지 하중과 정확히 경계가 겹쳐 여유 0 무릎이 통과됐음 (quad_0001 실측
   13.4Nm vs 정적 요구 12.3Nm). 새 규칙 = 지지 하중 F(m g / 절반 다리 수:
   quad 트롯 2·hex 트라이포드 3·humanoid 단일 지지) x 모멘트 팔 x 여유율
   U(1.25,3.0). 팔 = max(기립 자세 수평 팔(닫힌형: mammal 무릎 = 아래 세그 합 x
   sin|gamma-psi| 등), arm_min_frac 0.35 x 관절 아래 도달 길이 — 보폭 자세 대비)
2. **좌우 미러링**: 배치 지터·마운트 오프셋·관절 한계를 열(row) 단위 1회 샘플
   -> 좌우 동일 적용. roll/yaw 축 가동 범위는 우측에서 상·하한 마진 교환 (xz평면
   반사). humanoid는 기하가 원래 대칭이라 한계 샘플만 역할 단위 캐시로 미러.
   plan/urdf.md의 "다리 장착 위치 비대칭 지터" 축은 폐기 (레퍼런스 3편 전부 대칭
   유지 + 비대칭이 평균 정책 보행 해를 없애는 것 분석 확인 — plan 변경 사항)
3. **속도 한계 연동**: 종전 U(5,15) 무연동 -> 하한 = vel_margin 1.3 x 스윙 피크
   요구 (pi x v_max / 다리 전장, duty 0.5 사인 근사. v_max = froude x
   sqrt(g x 기립고), froude는 rl.yaml 명령 상한 규칙과 정합: multileg 0.6 /
   humanoid 0.35). 상한 = max(vel_hi, 1.6 x 하한). ANYmal 대입 시 8.1 rad/s로
   실값(8-10)과 정합 확인

- validate.py `_check_stance_torque` 추가 (이중 확인): 접촉 링크가 continuous 없이
  revolute로만 base와 연결되면 다리로 판정 (wheeled 자동 제외) -> 절반 지지 GRF를
  접지면 압력 중심(최저점 밴드 평균 — 꼭짓점 1개는 발 박스 모서리 과대 팔)에 걸어
  체인 관절 토크 x stance_torque_margin 1.2 <= 한계 검사. metrics에
  stance_margin_min/stance_worst_joint 기록
- 파일럿 재생성 (8 form x 3 + go2_proxy): 생성 전부 통과 (평균 시도 1.0-1.3,
  기각 루프 없음), 재검증 25/25, 렌더 24장 시각 확인. go2_proxy도 새 스탠스 검사
  통과 (공식 토크 여유 — 검사 눈높이가 실로봇과 정합한다는 신호)
- 검산 (URDF 재파싱): 좌우 토크·속도 완전 일치, roll/yaw ROM 정확히 부호 반전
  교환, stance_margin_min 1.59-2.85 (quad) / 2.85-13.7 (humanoid), 다리 속도
  하한 5.8-9.2 rad/s (규칙값 이상 확인)

## 로봇당 정책 학습 파이프라인 (tools/02_train_legged.py) — 스톡 go2 rough 태스크 + 로봇 교체

- 구조: RL 설정(관측·보상·종료·커리큘럼·PPO)은 Isaac Lab 내장 UnitreeGo2RoughEnvCfg
  그대로 + go2 표준 오버라이드 3건 동일 적용. 교체분 = 로봇 articulation(USD·
  DCMotor·게인)과 로봇 종속 참조(베이스 링크·발 링크 목록·토크 페널티 스케일).
  train / play(고정 명령 비디오: fwd·lat·stairs — 정책 없으면 zero-action 홀드
  진단) 서브커맨드, GPU 선택 = CUDA_VISIBLE_DEVICES, 로봇당 프로세스 1개 병렬
- 정책 출력: /data/EA-Trav/sim/policies/{form}/{이름}/ (무누적 — 학습 후 중간
  체크포인트 삭제, 최종 model + jit policy.pt + params만)

### 1차 학습 실패 -> 스케일 결함 2건 수정 (30분 만에 진단·중단)

- 증상: 에피소드 1-5s(base 접촉 100%), 보상 0 고착, 심각도가 질량 순서와 일치
- 결함 1 (스폰 붕괴): 게인 질량 비례 kp = 2 x m은 소형(15-50kg) 실로봇 대역의
  우연 — 대형 장다리(스탠스 요구 200Nm급)에서 평형 처짐 > 0.5 rad. **스탠스 토크
  앵커로 교체: kp = 5 x 관절별 스탠스 요구** (생성기가 meta params.stance_torque로
  기록 — multileg·humanoid 다리. go2 대입 18.5 vs 실값 25, H1 무릎 300 vs 200,
  ANYmal 150 vs 80 — 전 스케일 동일 자릿수 재현). kd = 0.025 kp
- 결함 2 (보상 스케일): 스톡 토크 페널티 -2e-4 x tau² 는 go2 절대 스케일 튜닝 —
  토크 수백 Nm급에서 추적 보상을 수십 배 압도. **go2 등가 정규화: 가중 =
  -2e-4 x (23.5 / 평균 토크 한계)²** ((tau/한계)² 정규화와 동치, 보상 함수는 스톡 유지)

### zero-action 홀드 진단 (계측 방법론) -> mammal 3절 기립 결함 발견·수정

- 진단: 평지 4 env + zero action 200스텝, 관절 그룹별 오차/토크/속도 + base 피치
  + 발·무게중심 월드 좌표 시계열 (관절이 버티는데 몸이 기우는지 = 기하 문제 판별)
- 판별: go2_proxy 안정(-12.8도 처짐 유지 = 하네스 무죄), sprawl 안정, **mammal
  3절만 전도·요동** — 발목 0도의 무릎-발목 일직선 기립이 좌굴 특이 구성
- 수정: mammal 3절 기립을 지행(digitigrade) 지그재그로 — 마지막 세그먼트 수직
  (IK는 l1·l2 2링크에 목표 h-l3, 발목 = gamma - psi. 주의: 조립이 관절각에
  knee_dir을 곱하므로 부호는 gamma-psi가 정답 — psi-gamma는 기울기가 2배가 되어
  지상고·셀프충돌·발목 스탠스 기각 폭증(시도 1.3->3.0)으로 드러남)
- 수정 후 홀드: quad_0001 0.3도 / quad_0002 0.0도 / go2_proxy 안정.
  quad_0000(72kg·기립 1.2m 극단 개체)은 여전히 38도 — plan의 동적 검사(평지
  기립) 탈락 개체로 분류, 학습 제외. **동적 검사 게이트(학습과 같은 액추에이터
  설정의 홀드 필터)를 full 생성 파이프라인에 편입 필요** (특이사항)
- 부수 수정: joints.json에 base_link(루트 링크 이름) 기록 (robot_spawn.convert_robot)
  — go2 계열은 "base"라 하드코딩 참조가 깨지는 것 실측. 02_train_legged가 사용
### 결과: 생성 로봇 보행 성립 (로봇당 정책 3/3)

- 3000 iter x 4096 env, 로봇당 GPU 1개 병렬, 약 2.5시간:
  go2_proxy(대조군) 보상 26.7 / quad_0002(mammal 3절 지행, 46kg) **27.4 —
  대조군 상회, 에피소드 만주** / quad_0001(sprawl, 69kg) 14.9 고원 (에피소드
  19s 안정 — 보행 성립하나 추적 효율 낮음, sprawl 걸음 특성 관찰 항목)
- 고정 명령 비디오 (fwd/lat/stairs x 3대, check/02_controller/legged/):
  3대 전부 기립 보행·명령 방향 추종·계단 씬 통과 시각 확인
- 판정: 수정된 생성기(대칭·스탠스 토크·속도 연동·지행 기립) + 스톡 태스크
  로봇 교체 파이프라인으로 **생성 URDF 로봇의 결정적 보행이 처음으로 성립**.
  스케일 적응 2건(게인 스탠스 앵커·토크 페널티 정규화)이 관건이었음
- 특이사항: play(비디오)에서 env 원점 랜덤 배정이 중복돼 로봇 2대가 겹쳐 보일
  수 있음 (물리 간섭 아님 — 시각만. 필요시 env 수 < 셀 수 + 중복 제거 검토).
  정책 산출물 = /data/EA-Trav/sim/policies/quad/{이름}/ (model_2999 + policy.pt
  + params). 다음: 동적 검사 게이트 편입 -> hex·나머지 quad 확장 -> humanoid는
  H1 계열 태스크 검토 -> 3단계(trav_gt) 롤아웃 연결

### 시각 불안정(요동) 개선 실험 — kd 상향 기각, 학습 연장 팔 진행 중

- 1차(3000 iter) 지표 대조로 원인 후보 정리: URDF 좌우는 완전 대칭 전수검산
  (기하·질량·한계·자세 불일치 0 — 걸음 비대칭은 RL의 대칭 자발 붕괴, 로봇 무죄).
  quad_0002는 대조군 동급(track 1.32, 부드러움 지표 개선 중 — 연장으로 해소 성격),
  quad_0001(sprawl)은 후반 action_rate 악화(-0.12 -> -0.18)·수직 출렁임 3배
- 발 마찰 레버 철회: 스톡 velocity env가 startup 이벤트로 전 로봇 공통 마찰
  0.8/0.6을 PhysX API 레벨에서 이미 적용 (physics_material EventTerm — 바인딩
  함정과 무관하게 동작. "유효 마찰 0.5" 추정은 이 env에선 오류였음). URDF에 없는
  물리는 전 로봇 공통 상수가 원칙 (plan 정합 — 로봇별 마찰은 은닉 변수를 만들어
  URDF=embodiment 전제 오염)
- kd_ratio 0.025 -> 0.05 A/B (go2_proxy 대조군 포함): **기각** — 1400-1700 iter
  시점 quad_0001 보상 하락(3.1 vs 기존 13.7), 대조군 절반(12.5 vs 24.3), 2체크포인트
  연속 악화로 조기 중단 (GPU 2.5h 절약). 기전 = DCMotor에서 감쇠 토크가 포화
  예산을 잠식 -> 감쇠 상향 = 가용 토크 손실. kd 0.025 원복
## 보행 품질 3계열 테스트 (사용자 지시 — "보행로봇답게 걷는가"를 지표·실험으로)

- 배경: 능력 지표(보상·지형 레벨)는 통과만 재고 걸음 품질을 못 잡는 것 확인
  (사용자 비디오 판정과 지표의 괴리). 보행 안정성 체크(지표)와 개선 실험을 분리
- **보행 품질 지표 6종** (play 비디오 롤아웃에서 동시 측정, 02_train_legged play가
  check/02_controller/legged/gait_metrics.json에 병합 저장):
  falls / slip_per_m(접지 중 발 수평 이동 ÷ 몸체 경로) / cot(sum|tau qdot|dt ÷ m g 경로) /
  contact_time_var(체공·접지 분산 — 절뚝임) / diag_async(대각 쌍 접촉 불일치 — 트롯 위상) /
  vz·wxy_rms(출렁임). 합불 앵커 = go2_proxy 측정값 (상대 기준, 크기 무관)
- **베이스라인 (fwd 0.8m/s, 현행 정책)**: go2_proxy slip 0.079/CoT 0.77/var 0.028 vs
  quad_0001 slip 0.172(2.2배)/CoT 2.55(3.3배)/var 0.074(2.7배), quad_0002 slip 0.148/
  CoT 1.04/var 0.060 — 사용자가 본 "허둥댐"의 정량화 성립
- **실험 팔** (--arm 플래그, 팔별 접미사 폴더 -> 판정 후 승자 승격·패자 삭제):
  - gait = Isaac Lab 내장 Spot 레시피의 걸음새 항 이식 (GaitReward 대각 위상 강제
    3.0 + air_time_variance -0.3 + foot_slip -0.15 + base_motion -0.6. 가중 스케일
    0.3 = go2/Spot 추적 가중 비율. foot_clearance는 절대 z 목표 = 평지 전용,
    base_orientation은 경사 정렬 상충이라 제외. 대각 쌍은 접촉 링크 이름에서 자동
    유도 — leg{row}{l|r} / F|R L|R 두 규약 지원, 4족 전용)
  - energy = 기계 일률 |tau qdot| 페널티 (Fu et al. 계열, 가중 = -0.001 x
    (A1 토크 33.5 / 평균 한계) 스케일 정규화)
  - AMP(모션 모방)는 생성 로봇마다 참조 모션이 없어 구조적으로 부적합 -> 배제
    (설치본에 4족 AMP 태스크도 없음 — 외부 IsaacGym 계열 필요)
- 라운드 1 (GPU 0-2, 5000 iter): gait x quad_0001 / energy x quad_0001 / gait x
  quad_0002 — 판정은 보상이 아니라(팔마다 항이 달라 비교 불가) 위 지표 + 비디오
- **라운드 1 결과 (fwd 0.8m/s 지표, base 대비)**:
  - quad_0002 x gait: **걸음 품질 대폭 개선** — 접촉 분산 0.060 -> 0.010 (5.7배),
    대각 비동기 0.253 -> 0.066 (3.8배 — go2 base보다도 동기적), 미끄럼 동등,
    이동 정상 (path 7.6m). 대가 = CoT 1.04 -> 2.17 (위상 강제의 에너지 비용)
  - quad_0001 x gait: **실패 (제자리 걸음 국소해)** — path 0.81m/10s (추종 붕괴,
    발만 동기적으로 구름). gait 가중 3.0이 추적(1.5+0.75)을 압도 + sprawl의 낮은
    추적 능력 조합. 프레임 확인: 계단 스폰 지점에서 정지 상태 스텝핑
  - quad_0001 x energy: **실패 (전면 악화)** — 미끄럼 0.17 -> 0.42, CoT 2.5 -> 3.2,
    비동기 0.23 -> 0.39, 계단 전도 1. 에너지 페널티 단독은 sprawl 걸음을 못 고침
  - 사고 기록: 비디오 파일명에 팔 미포함 -> 병렬 팔 촬영이 서로 덮어씀 (지표
    json은 팔별 키라 무사). 파일명 규약을 {로봇}__{씬}__{팔}로 수정 후 재촬영
- 라운드 1 판정: 위상 강제(gait)는 트롯 호환 형태(mammal 지행)에서 유효, sprawl은
  가중 재조정 필요 (후보: gait 1.5로 반감 — Spot 원 비율 1.0에 근접). 에너지 팔 기각

- 연장 팔 (kd 0.025 x 5000 iter) 판정 — **5000 iter를 로봇당 학습 표준으로 채택**:
  - go2_proxy 26.7 -> 29.1 (track 1.28 -> 1.33, action_rate -0.120 -> -0.111 개선)
  - quad_0002 27.3 -> 28.9 (track 1.32 -> 1.36 — 대조군 동급 유지, 요동 동일)
  - quad_0001 15.1 -> 17.1: 지형 레벨 5.4 -> 5.8 (능력 향상 — 계단 등반 비디오 확인),
    단 track 1.076 정체 + action_rate -0.199 잔존 = **요동은 학습량·감쇠로 해결
    안 되는 sprawl 형태 고유 잔차로 확정** (연장·kd 두 팔 모두 기각됨).
    GT 관점에서는 통과 능력(지형 레벨)이 정상 대역이라 수용, 필요시 다음 후보 =
    부드러움 페널티(action_rate·dof_acc)의 형태별 스케일 규칙
  - 산출물 갱신: /data/EA-Trav/sim/policies/quad/{3대}/ (model_4999 + policy.pt),
    고정 명령 비디오 9편 재촬영 (구판 교체). GPU 사용 규칙: 이후 0-2만 (사용자 지시)
