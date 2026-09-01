# 2026-09-01

- **wheeled 컨트롤러 구조 개편**: `2D waypoint -> PP -> body cmd` 단일 단계를
  `planner(MPPI) -> pursuit(PP) -> governor(CBF) -> ik` 4단계 파이프라인으로 교체 — 경사 지형을
  전혀 못 보던 구조를 고치기 위함. 네이밍 규칙: 역할이 드러나는 이름 우선, `core/`(범용 알고리즘)는
  `{역할}_{알고리즘}.py`, `wheeled/`(로봇별 구체화)는 역할 이름만.
  - `core/scan_terrain.py`(신규): raycaster 히트를 로컬 높이/경사 격자(`TerrainScan`)로 재구성 +
    쌍선형 보간. `ray_alignment="world"`(축정렬, wheeled 전용) vs legged의 `"yaw"`(자기중심)를 구분.
  - `core/plan_mppi.py`(신규): 범용 MPPI 샘플러 — dynamics_fn/cost_fn을 주입받아 로봇 결합 없음.
  - `core/govern_cbf.py`(신규): CBF 닫힌 해 투영 유틸(`project_interval`/`project_halfplane`) —
    barrier가 1-2개뿐이라 cvxpy 등 QP 솔버 없이 batched torch로 처리 (GPU 병렬·의존성 추가 없음).
  - `core/high_pp.py` → `core/track_pursuit.py` 이름만 변경(로직 무변경, wheeled/legged 공용 유지).
  - `wheeled/low_ik.py` → `wheeled/ik.py` 이름만 변경.
  - `wheeled/planner.py`(신규): MPPI wrapper — unicycle/bicycle/holonomic 롤아웃 동역학 + 목표거리·
    지형경사·control effort 비용. 출력은 전체 경로가 아니라 최적 궤적 위 lookahead 지점 1개
    (PP 인터페이스를 그대로 재사용하기 위한 의도적 축소).
  - `wheeled/governor.py`(신규): rollover 방지 CBF — [Rollover Prevention for Mobile Robots with
    CBF (arXiv 2403.08916)]의 ZMP 기반 barrier(`h = ±v·w - tip_accel·gravity_b_z ∓ 9.81·gravity_b_y`)를
    구현. `RobotCtrlParams.tip_accel`(`= g·b/l_cg`)이 논문의 b/l_cg와 같은 양이라 그대로 재사용 —
    기존 `controller.py`의 즉석 tip_accel 속도 캡(평지에서도 상시 적용되던 사전예방형)을 대체,
    실제 자세(gravity_b) 기준으로 필요할 때만 개입하는 실시간 barrier로 교체.
  - `core/base.py`: `ControlObs`에 `terrain_scan` 필드 추가(TYPE_CHECKING 지연 임포트로 isaaclab
    런타임 의존성 없이). `rollout.py`: wheeled용 world-aligned 지형 스캐너 연결(legged와 동일
    패턴, `scene != "flat"`일 때만), scanner_cfg 빌더를 legged/wheeled 공용으로 정리.
  - 단위 스모크 테스트(`core/plan_mppi`, `core/scan_terrain`, `core/govern_cbf`,
    `wheeled/governor`, `wheeled/planner`) 전부 실측 통과. governor 최초 구현에 9.81 중복 곱
    버그 있었음(발견·수정·재검증 완료) — `tip_accel`이 이미 `g·b/l_cg`라 g_z항엔 9.81을 또
    곱하면 안 되고, g_y항에만 곱해야 함.
  - **실측 회귀 발견·수정**: 실제 Isaac Sim 평가(diff_0000)에서 flat 시나리오가 첫 웨이포인트
    도달 후 완전히 정지하는 회귀 확인. 원인: `LocalPlanner`의 MPPI 롤아웃 dt를 제어 주기(0.02s)와
    같게 뒀더니 horizon(12스텝)을 다 더해도 0.24초만 내다봐서, lookahead 지점이 로봇 반경
    0.15m 안에 갇히고 PP의 감속 로직(`v_stop`)이 상시 "곧 도착"으로 오판 — 한번 정지하면 이동
    거리가 0이라 다음 재계획도 같은 좁은 반경에 갇히는 자기강화 루프. 수정: planner의 dt를
    제어 주기에서 분리(`mppi.plan_dt=0.15`, 실제 초 단위)하고, MPPI 자체는 제어 루프보다
    낮은 주기로만 재계획(`mppi.replan_decimation=10`)하도록 `controller.py`에 캐싱 추가 —
    그 사이는 PP가 원래 고주기 추종으로 채움. 실측으로 재검증: flat 3/3 도달 정상화.
  - slope 전복 1차 가설(MPPI 재계획의 확률적 흔들림) 기각 후, 이미 저장된 debugging_trace를
    재분석(추가 시뮬 없이)해 진짜 패턴을 확인: 로봇이 x≈1.48-1.5m 부근에서 root_z 변화 없이
    (아직 평지) v≈0으로 거의 멈춘 채, cmd_w가 몇 스텝 만에 0→-1.0대로 급격히 커지고 그만큼
    바퀴 토크가 크게 비대칭(예: 2.48 vs -0.27)으로 실리면서 넘어짐.
  - 원인 하나 확정·수정: governor의 w 구간(`tip_accel/v`)이 v가 작을수록 넓어져 v→0에서
    사실상 무제한이 되는 설계 허점 — 마침 로봇이 거의 멈춘 순간에 barrier가 사실상 뚫려있었음.
    `wheeled/governor.py`에 `v_max`(설계 순항 속도) 기준 절대 상한(`w_cap = tip_accel/v_max`)을
    바닥으로 깔아, 저속에서 순항 속도보다 더 자유로워지는 구멍을 막음(`near_zero` 우회 로직도
    제거 — w_cap 도입으로 v=0 특이점이 자연히 처리됨). 실측 재검증: flat 3/3 유지, slope 전복
    시각이 2.92s -> 3.27s로 다소 늦춰졌으나 여전히 발생 — 이 수정은 옳지만 근본 원인은 아니었음.
  - **미해결**: 전복 시점의 cmd_w(~-1.0~-1.16 rad/s)가 w_cap보다 한참 작아 governor가 애초에
    관여하지 않는 상태에서도 넘어짐 — ZMP-v*w 모델 자체가 못 잡는 물리(급가속 시 바퀴 토크
    반작용/슬립)이거나, PP+PID 루프가 v→0 부근에서 불안정한 별개 문제로 보임. 로봇이 매번 같은
    x(~1.48-1.5m)에서 거의 멈추는 것도 우연이 아닐 가능성 - MPPI가 전방 스캔으로 앞쪽 경사를
    미리 감지해 `slope_weight` 비용으로 진행을 억제하다 "가야 하나 말아야 하나" 경계에서
    속도가 0에 가까워지고, 그 상태에서 PP/PID가 작은 오차에도 과도한 w를 내는 조합으로 추정
    (미확인). 다음 확인 필요: (1) local_goal·terrain_scan 값을 실제로 찍어 이 경계 거동 확인,
    (2) `yaw_accel`(현재 4.0, 이 로봇에 비해 과할 수 있음)이나 `mppi.slope_weight` 재튜닝,
    (3) v가 작을 때 w 자체도 같이 눌러주는 저속 안전장치 추가 — 방향 선택은 사용자 확인 필요.

# 2026-08-30

- `tools/01_sim.py`: standing-height/tilt 판정 보고서가 새 스키마임을 드러내도록 `schema: 01_sim.standing_height_tilt.v1` 추가.
- `tools/02_controller.py`: CLI `--device`와 실제 `SimEnvironment`/controller/PPO train device가 엇갈리던 경로 수정. 이제 `args.device`가 eval/train 전체에 전달됨.
- `scripts/urdf/utils/graph_parser.py`: URDF 인코더 입력용 그래프 파서 추가. fixed/mimic child는 merge하고 `*_roller_*` 링크는 부모 wheel 노드의 `roller_summary`로 fold.
- Omni roller parser smoke:
  - `omni_0000`: raw_links 52 -> graph_nodes 4, rollers folded 48
  - `omni_0001`: raw_links 53 -> graph_nodes 5, rollers folded 48
  - `omni_0002`: raw_links 46 -> graph_nodes 4, rollers folded 42
- Hidden wheel test:
  - 정적 collision 검사에서 `wheel_exposed=false` 샘플은 base-wheel collision overlap이 실제로 존재함.
  - Isaac standing: `diff/diff_0000` PASS, `skid/skid_0000` PASS.
  - Controller flat: `diff/diff_0000` OK 3/3, `skid/skid_0000` OK 3/3.
  - Controller slope: `diff/diff_0000` FAIL 0/3, final_dist 4.051m, max_tilt 9.2deg. `skid/skid_0000` FAIL 0/3, final_dist 4.441m, max_tilt 0.3deg.
  - 결론: hidden wheel은 몸통-휠 관통 설계이지만 즉시 스폰/기립/평지 주행을 깨지는 않음. slope 실패는 hidden collision 폭발보다 현재 wheeled controller terrain 성능 문제로 보는 쪽이 맞음.
- 검증:
  - AST syntax OK: `tools/01_sim.py`, `tools/02_controller.py`, `scripts/urdf/utils/graph_parser.py`.
  - `py_compile`은 코드 문법이 아니라 `tools/__pycache__` 권한 문제로 실패.
- RL legged 분석 시작:
  - 현재 `tools/02_controller.py eval`은 `policy.pt + bundle.json`이 있는 새 form 정책만 평가한다. 워크스페이스/`/data`의 기존 legged 정책은 `bundle.json`이 없어 현재 eval에서 정상적으로 스킵된다.
  - `gait_metrics.json`과 `/data/EA-Trav/sim/policies/quad/*`는 구 경로 `tools/02_train_legged.py` 산출물이다. 새 경로(`tools/02_controller.py train`)와 직접 호환되지 않는다.
  - 구 경로는 generated quad 게인을 `stance_torque x 5`로 잡지만, 새 form-bundle 경로의 `low_rl.loco_gain_overrides`는 아직 `kp_per_kg x mass`만 쓴다. 예: `quad_0001` 새 경로 kp 137, stance-torque 기준 221-1592. full/multi-robot로 갈 때 이 불일치가 재발 가능성이 큼.
  - pilot 설정은 `go2_proxy` 1대, `terrain_mode: flat`, `include_morph=false`, `include_scan=false`라 현재 목표는 멀티형태/험지가 아니라 go2_proxy 평지 보행 창발 검증이다.
- Wheeled slope datalog:
  - `tools/02_controller.py eval`에 `--datalog` 옵션 추가. 저장 위치는 `check/02_controller/{wheeled|legged}/datalog/{rel}__{scene}.json`.
  - datalog 샘플은 root pose/velocity/tilt, command, wheel velocity/target/torque, wheel effort limit 대비 torque ratio를 기록한다.
  - `diff/diff_0000` slope FAIL: mean cmd vx 1.086m/s, mean body vx 0.044m/s, mean wheel target 13.955rad/s, mean wheel velocity 13.844rad/s, mean torque ratio 0.021, peak 0.333. 바퀴는 목표 속도를 거의 따라가지만 차체 진행이 멈추므로 단순 토크 부족보다 terrain 접촉/기하 coupling 문제가 더 큼.
  - `skid/skid_0000` slope FAIL: mean cmd vx 0.963m/s, mean body vx 0.029m/s, mean wheel target 12.031rad/s, mean wheel velocity 10.053rad/s, mean torque ratio 0.966, peak 1.0. hidden skid는 slope에서 토크 포화로 stall.
  - `skid/skid_0002` exposed slope OK: mean cmd vx 1.066m/s, mean body vx 0.680m/s, mean wheel target 21.715rad/s, mean wheel velocity 21.033rad/s, mean torque ratio 0.394, peak 1.0. 같은 skid 계열에서 exposed wheel은 slope 통과.
  - 결론: slope 실패는 hidden wheel 하나로만 설명되지 않는다. hidden skid는 물리적으로 불리한 URDF/구동 조건이 직접 관찰되지만, diff hidden은 wheel spin이 body motion으로 연결되지 않는 접촉/기하/terrain 상호작용 문제가 우선이다. wheeled terrain 평가는 controller, traction/friction, collision geometry, actuator limit을 함께 보는 datalog 기반 판정이 필요하다.
- RL/X-Nav reference check:
  - X-Nav는 random embodiment expert DRL + distillation 구조이며, expert observation에 proprioception, last action, reference speed, goal, time-left, privileged embodiment parameters, terrain height scan을 포함한다.
  - quadruped expert는 gait timing observation을 사용하고, reward는 additive sum만이 아니라 `r_task * exp(c_reg * r_reg)` 형태로 task/regularization을 결합한다.
  - X-Nav quadruped regularization에는 collision, joint acceleration, vertical base velocity, roll/pitch angular velocity, torque, action rate, joint deviation, action acceleration, swing phase tracking, stance phase tracking, foot sliding avoidance가 들어간다.
  - 현재 pilot/new RL 설정은 flat, go2_proxy 중심, morph/scan/goal-time/gait phase가 빠져 있고 `undesired_contact=0`이라 lower/shank dragging을 적극적으로 배제하지 않는다. 사용자가 지적한 3족 지지+1족 들림 문제는 접촉 패턴 reward 하나를 덧붙이는 문제가 아니라, foot contact body 정의와 locomotion regularization/environment 구성이 너무 빈약한 문제로 보는 것이 맞다.
- Wheeled slope 실패 원인 진단 (임시 스크립트 2종, `_tmp_diag/` — 분석 후 삭제):
  - 가설 1(마찰 바인딩 결손) 확인: `tools/01_sim.py`는 `robot_spawn.standard_friction_links(meta.contact_links)`를 `friction_links`로 넘겨 구동 바퀴에 표준 마찰(1.0)을 명시 바인딩하지만, `tools/02_controller.py`의 `_run_one`은 이 호출을 legged 폼(`slots.contact_links`)에만 적용하고 wheeled 구동 바퀴에는 적용하지 않는다. 스테이지 감사로 실측: baseline은 `wheel_l`/`wheel_r` 물리 재질이 `None`(PhysX 기본 마찰로 방치), 표준 바인딩 적용 후엔 `static=dynamic=1.0`. 이 코드베이스에서 이미 2회 기록된 결함 부류(`environment.bind_link_friction` 문서)의 재발 — **버그는 실재하므로 `_run_one`도 `standard_friction_links`를 wheeled에 적용해야 함**.
  - 단, 마찰 수정만으로는 slope FAIL이 해소되지 않음 (`diff/diff_0000` final_dist 4.051→4.055m, `skid/skid_0000` 4.441→4.444m — 사실상 불변). `skid_0000`는 마찰 수정 전후 모두 peak torque ratio 1.0(포화) — 액추에이터 한계가 별도 원인으로 재확인.
  - **근본 원인 확정 (ContactSensor 직접 측정)**: `diff_0000`/`skid_0000`(hidden wheel) 바퀴·base_link에 `PhysxContactReportAPI` + `ContactSensor`를 달아 개방루프 구동(펄스 pursuit 배제, 바퀴 고정 각속도 10rad/s)으로 측정. 평지에서는 바퀴 접촉력이 정상(30-150N, base_link 접촉 0)이고 정상 주행(vx 0.77m/s)하지만, slope 진입 후(diff: t=3.0-3.6s, tilt 0→7.3°) **바퀴 접촉력이 완전히 0으로 떨어지고 base_link 접촉력이 대신 47-51N으로 발생, 바퀴는 계속 10rad/s로 도는데 차체는 정지**(vx≈0, tilt 6.95°에서 영구 고착). skid도 동일 패턴(base_link 접촉력 184N, 바퀴 대부분 0).
  - 결론: "hidden wheel"(`wheel_exposed=false`, 바퀴가 몸통 안쪽으로 숨는 배치)이 경사에서 몸통 콜라이더의 실효 지상고를 깎아, 바퀴보다 몸통 바닥이 먼저 지면에 닿아 "얹힌"(high-centered) 상태가 된다. 이후 바퀴는 완전히 뜬 채 헛돌고, 몸통은 마찰만으로 버티며 정지 — 토크 비율이 낮은데도 못 나가는 diff 패턴과 토크 포화되는 skid 패턴 둘 다 이걸로 설명됨. 마찰 계수 수정이 안 먹힌 이유도 동일 — 하중을 받는 게 애초에 friction_links 대상이 아닌 base_link였기 때문.
  - 원인은 controller 로직이 아니라 **URDF 생성기의 정적 지상고 검사가 기립 자세(평지)만 검증하고 피치(경사) 상태의 유효 지상고는 검증하지 않는 것** — `plan/urdf.md` 정적 검사 항목(지상고·셀프충돌·접지 공면 등)에 "기울어진 자세에서의 지상고" 축이 없음. 수정 방향은 (a) hidden wheel 지상고 여유(margin)를 경사 목표각 이상 확보하도록 생성기 제약 강화, 또는 (b) 정적 검사에 피치 자세 지상고 검사 추가 — 둘 다 00_urdf 단계 변경이 필요해 사용자 확인 후 진행.
- Wheeled 마찰 바인딩 결손 수정 (`tools/02_controller.py` `_run_one`): `friction_links`를 legged(`slots.contact_links`)일 때만 채우던 것을, wheeled는 `robot_spawn.standard_friction_links(meta["contact_links"])`로 채우도록 분기 추가 — `01_sim.py`와 동일 규약으로 정합. slope FAIL 자체는 근본 원인(hidden wheel high-centering, 00_urdf 쪽)이 남아 있어 해소되지 않지만, 이미 2회 재발한 마찰 미바인딩 결함 부류는 이걸로 제거. AST 문법 검사만 통과 (컨테이너 재실행 검증은 미실시).
- **hidden wheel 재진단 (대량 표본, GPU 4장 병렬, 임시 스크립트 — 조사 후 삭제)**: `wheel_exposed=false`가 원인이라는 위 결론을 diff/skid 각 15대(seed=777) 슬로프(0.2rad) 평가로 재검증. 3구간 회전 코스는 경사 위 급선회로 전도를 유발해 정체와 섞이므로 직진 단일 목표로 단순화.
  - **원 가설 기각**: diff에서 정체 실패 7건이 전부 `wheel_exposed=true`(exposed)였고 `false`(hidden) 4대 중 3대는 통과 — 원 결론과 반대 방향. 횡방향 노출 여부는 경사 통과와 무관.
  - **diff의 실제 변수는 수직 clearance/radius 비율**: 통과 7대 평균 0.84 (0.62-1.16), 정체 7대 평균 0.53 (0.33-0.69) — 대략 0.6-0.7이 경계. 실로봇 조사(아래) 구간과 방향 일치.
  - **skid는 이 변수로 설명 안 됨**: 15대 중 12대 통과, clearance_ratio 0.48(낮음)도 통과·1.13(높음)도 정체 — front/rear 오버행·임계각 모델(atan(clearance/overhang))도 diff는 어느 정도 맞지만(20/28, tipover 2건 제외) skid는 안 맞음. skid 정체 원인 미상 (표본 부족, final_dist 0.5m 내외라 물리적 정체인지 단순 시간초과인지도 불명확).
  - **별개로 발견한 전도(tip-over) 버그**: `diff_0005`(clearance_ratio 0.998)·`skid_0000`(1.232) — 지상고가 넉넉한데도 정체가 아니라 max_tilt 60도+ 로 전복. 시드 재현 후 비디오+datalog로 단독 재조사 완료.
    - 두 로봇 모두 **몸통이 풋프린트 대비 비정상적으로 높은 "타워형"** (diff_0005: body 0.181x0.181m에 높이 0.32m / skid_0000: body 1.02m 길이에 track폭 0.114m·높이 0.20m) — 비디오로 diff_0005는 경사 진입 직후 붕 뜨며 공중제비, skid_0000은 20초간 오르다 옆으로 서서히 전복되는 것을 육안 확인.
    - **diff_0005 개별 원인 (명확)**: `wheeled_base.py::_set_drive_limits`의 `v_max = rng.uniform(0.5, 3.0)`가 로봇 크기와 무관한 절대 m/s 범위 — 이 로봇은 max_lin_vel=2.43m/s로 몸길이(0.181m) 대비 초당 13.4배 속도가 목표 명령. datalog상 1.5초 만에 이 속도까지 가속 후 경사 진입 지점에서 튀어오르며 전복. 실로봇 대비(turtlebot3 burger 최고속 0.22m/s ≈ 몸길이의 1.5배/s) 명백히 비현실적 상대속도.
    - **skid_0000 개별 원인 (약함)**: 속도는 정상(2.85 몸길이/s)이지만 트랙폭(0.114m)이 몸길이(1.02m)·높이(0.20m) 대비 매우 좁음 — 롤 안정 여유가 애초에 얇았던 상태로 20초 등판 중 서서히 전복. 정적 검사의 "무게중심-지지 다각형" 항목이 이런 세장비를 왜 걸러내지 못했는지는 미조사.
  - 조치 방향 미정 (사용자 확인 대기): (1) `max_lin_vel`을 로봇 크기(체장·바퀴반지름) 비례로 재샘플링, (2) 정적 검사의 CoM-지지다각형 마진 강화 또는 세장비(높이/풋프린트) 상한 추가, (3) diff clearance_ratio 하한을 실로봇 구간(≥0.6 근방)에 맞춰 강화. 셋 다 `scripts/urdf/platform/*` 또는 `configs/urdf.yaml` 변경이 필요해 보류.
- **diff 전후 지지폭(wheelbase) 누락 수정 + 실측 검증**: `extract_ctrl_params`의 `tip_accel`(전도 한계 가속도) 계산이 diff 폼에서 `track_width`(좌우)만 보고 전후(피치) 지지폭을 아예 안 봤던 결손을 수정. `scripts/urdf/platform/diff.py::sample`에 캐스터·구동축 배치 직후 `spec.params["wheelbase"] = 2 * min(무게중심에서 축·각 캐스터까지 거리)`를 추가 — track_width처럼 좌우대칭을 가정할 수 없어(diff는 축 1개 + 편측 캐스터) `wheelbase/2` 근사 대신 무게중심 기준 실제 최소 거리를 직접 계산.
  - 검증 1 (분포): seed=777로 diff 30대 재생성 — 정적 검사 통과율 불변(30/30), 전 개체에서 `tip_pitch`(신규)가 `tip_roll`(기존)보다 훨씬 작게 나옴(예: diff_0000 roll 11.99 vs pitch 1.18) — 원래 계산이 훨씬 느슨한 축만 보고 있었다는 진단이 개별 사례가 아니라 diff 폼 전반의 구조적 결손이었음을 확인.
  - 검증 2 (재현): `diff_0005`(원래 전복 개체) 재평가 — cmd 가속 램프는 확실히 느려졌지만(t=1.0 cmd 1.47->1.18m/s) **여전히 전복**(tilt 61.3, t=2.3s). datalog 확인 결과 `lin_accel`(가속 램프)만 낮아졌을 뿐 `v_max`(정속 상한)는 tip_accel 보정을 안 받아 그대로 2.43m/s — 결국 같은 절대속도까지는 도달해 경사 턱에서 동일하게 튐. **이 수정은 필요하지만 단독으로는 diff_0005급을 못 막는다** — 위 조치 후보 (1)(max_lin_vel 크기 비례 재샘플링)이 같이 들어가야 함이 직접 실측으로 확인됨.
  - 검증 3 (대조군): `diff_0021`(원래 roll이 더 타이트한 케이스)은 tilt 11.3도로 깨끗하게 목표 도달 — 수정이 정상 케이스를 망가뜨리지 않음. `diff_0011`(tip_combined 0.83, 이번 배치 중 가장 타이트)은 도달은 했으나 tilt 56.1도로 아슬아슬 — 가속 캡만으로는 여전히 여유가 얇음.
  - AST 통과, 컨테이너 내 실제 생성·USD 변환·시뮬 평가(영상+datalog)로 검증. 임시 스크립트·데이터는 조사 후 삭제 완료.
- **max_lin_vel 체장 비례 재샘플링 + 실측 검증**: `wheeled_base.py::_set_drive_limits`의 `v_max = rng.uniform(0.5, 3.0)`(절대 m/s)를 `rng.uniform(0.5, 3.0) * spec.params["body_length"]`(체장 비례 배/s)로 변경. `_build_base`가 `body_length`를 항상 먼저 기록하므로 시그니처 변경 없이 diff/skid/ackermann/omni(x2)/wheeled_humanoid 6개 호출부 전부에 적용됨.
  - 검증 1 (분포): seed=777로 diff/skid/ackermann/omni 각 15대 재생성 — 정적 검사 통과율 불변(60/60), 4개 폼 전체에서 몸길이 대비 속도비가 정확히 [0.5, 3.0] 범위 안에 들어옴 (수정 전 diff_0005는 13.4배였던 것이 이제 최댓값 2.96배로 강제 상한).
  - 검증 2 (재현, wheelbase 수정과 함께): `diff_0005` 재평가 — max_lin_vel 2.43->0.44m/s로 실측 확인. datalog상 cmd가 t=0.4s부터 0.397m/s로 완전히 평평하게 유지되며 t=4.6s(주행거리 1.78m)까지 tilt 0.02도 이내로 안정 주행 — **가속·정속 문제는 실측으로 완전히 해소됨**. 그런데 t=4.7s부터 갑자기 tilt가 치솟아(1.6->52도) **동일하게 전복**(t=5.5s, tilt 60.6). 원인은 속도가 아니라 위치 — 경사 시작 지점(중앙 평지 platform_width=2.0m, 가장자리 x=1.0m)을 한참 지난 x≈1.78m 지점에서 일어남. **세 번째 원인 미상**: 정속 저속 순항 중에도 이 로봇(바퀴반지름 5.25cm, 몸통 0.18x0.18x0.32m 타워형)만 경사면 어딘가에서 넘어짐 — heightfield 지형 메시의 국소 요철/단차이거나 이 극단적 형상 고유의 문제일 수 있음, 미조사.
  - 검증 3 (대조군, 앞 항목과 동일 로봇): `diff_0011`은 결과 불변(도달하지만 tilt 56.0도로 여전히 아슬아슬). `diff_0021`은 재평가 중 컨테이너 프로세스가 원인 불명으로 무응답 종료(에러 로그 없음, 이 수정과 관련 있는지 불명) — 재확인 안 함.
  - AST 통과, 컨테이너 내 실제 생성·시뮬 평가로 검증. **결론: 두 수정 모두 각자 의도대로 정확히 동작함을 실측으로 확인했지만, diff_0005는 여전히 미해결 — 세 번째의 다른 원인이 있음.** 임시 스크립트·데이터는 조사 후 삭제 완료.
