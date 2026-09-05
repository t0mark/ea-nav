Embodiment-aware Traversability 연구를 진행할거야.
제안 방법 : URDF를 모델에 입력으로 추가해서 Embodiment에 맞는 Traversability 평가를 진행

### 연구 문제 설정

- 타겟 하드웨어 : wheeled 로봇, 다족보행 로봇, 휴머노이드 — 실제 로봇 URDF를 종류별로 여러 대 수집해서 사용
- 타겟 케이스
    - 계단 단차, 경사 각도
    - 좁은 폭, 책상 아래를 통과할 때 로봇의 물리적 크기 고려

### 모델 구조

- Depth → 인코딩 → 디코딩 → 이동가능성 헤드
- URDF → 그래프 인코딩 → 디코딩 단계에 주입
- 출력 : RGB 이미지 위의 히트맵 (가시성 때문에)
- Depth 인코더
    - 입력 전처리 : Depth → Intrinsic 역투영 → x,y,z 입력 (base_link 좌표계) → 고정 스케일 정규화
        ⇒ Ref: DUSt3R — 뎁스를 이미지 좌표가 아닌 3D 포인트맵(x,y,z)으로 표현하는 방식을 가져옴
    - 구조 : 패치 임베딩 (Scratch) + DINOv3 (LoRA)
        ⇒ Ref: RVT — RGB 사전학습 ViT에 비-RGB 입력(포인트맵)을 넣기 위해 패치 임베딩만 새로 학습하는 구성을 가져옴
- URDF 인코더
    - 입력 전처리 : URDF → 그래프 파서 → 토크나이저 (토큰 = concat(링크 필드, 부모 조인트 필드))
        - fixed joint 링크는 부모에 포함 (센서 등)
        - 노드 (링크) : 충돌 형상 타입 (one-hot), 형상 크기 파라미터 (3차원, 제로 패딩), 형상 로컬 오프셋 (자세 6D), 질량 (log-scale), base_link 플래그
        - 엣지 (조인트) : 조인트 타입 (one-hot), 축 (3), 원점 변환 (위치 3 + rpy 3), 가동 범위, 속도/토크 한계
        ⇒ Ref: MetaMorph, Amorpheus — 키네마틱 트리를 순회 순서로 펼쳐 링크 1개 = 토큰 1개인 시퀀스를 만들고, 각 토큰 피처에 해당 링크 속성 + 부모 조인트 속성을 concat하는 구성을 가져옴. 로봇마다 링크 수 N이 달라지는 문제는 패딩 + 어텐션 마스크로 처리
    - 구조 : 링크 토크 나이저 (부모 인덱스 + 트리 깊이 임베딩) + Self-attention (전체 Scratch)
        ⇒ Ref: Graphormer — 그래프를 Transformer로 처리할 때 그래프 거리 기반 어텐션 바이어스와 구조 임베딩을 쓰는 방식을 가져옴 (URDF 트리에 적용)
    - 출력 : CLS 벡터 1개 + 링크 피처 N개
- 디코더
    - 구조 : [self-attn → cross-attn → FFN] 블록 + 업샘플링 (전체 Scratch)
        ⇒ Ref: Flamingo — 메인 모달리티 토큰이 query, 조건 모달리티 토큰이 key/value인 cross-attention 주입 구조를 가져옴 (텍스트→이미지 자리에 이미지→URDF를 대입). 컨디셔닝이 모달리티 독립적이라 추후 RGB 전환에도 유지 가능
        ⇒ Ref: DPT — ViT 토큰에서 픽셀 단위 dense 출력을 만들 때의 단계적 업샘플링([bilinear ×2 + conv] 반복) 구조를 가져옴

### GT

- Traversability GT 생성 방법   
    - GT 정의 : 현재 위치에서 해당 위치까지의 도달 가능성, "이 embodiment + 제어기가 실제로 해당 지점까지 직진으로 통과할 수 있는가"
    - 파이프라인
        1. 그리드 맵 생성
        2. 통과 가능성 롤아웃 : 각 셀에서 로봇 스폰 → 통과 시도 (방향 x 반복 횟수) → 셀 점수 저장 (셀 점수 = 실패 : 0, 성공 : 안정성 점수)
            ⇒ Ref: Chavez-Garcia et al. — sim 롤아웃 성공/실패로 traversability 라벨을 만드는 파이프라인을 가져옴 (이진 → 연속으로 확장)
            ⇒ Ref: WVN — 명령 대비 실제 속도 추종 오차를 연속 traversability 점수로 쓰는 방식을 안정성 점수 지표로 가져옴 (기울기·미끄러짐 지표 추가)
        3. 카메라 포즈 랜덤 결정 + 뎁스 이미지 렌더링
        4. 히트맵 GT 생성: 뎁스 픽셀 역투영 → 2번 롤아웃의 셀 통과가능성 점수를 이미지에 바로 매핑, 서 있을 수 없는 표면은 ignore 마스크

### 학습

- 전체 학습 계획
    1. 로봇 URDF, USD 수집
    2. 제어기 생성 (legged = 로봇별로 RL 보행기 생성, wheeled = 타입별 기구학 + isaac sim ArticulationController)
    3. Sim 환경에서 이동하면서 위치별로 주행 가능 판단
    4. 3번을 GT로 사용하여 이동가능성 모듈 학습
    5. 입력 데이터는 랜덤 위치에서의 뎁스 이미지
- 학습 설정
    - 메인 손실 : 히트맵 BCE (soft target)
    - 보조 손실 : CLS → 로봇 물리 속성 예측 헤드 (전폭, 전고, 최대 단차 능력, 바퀴/다리 여부 등) — 라벨은 URDF 파서에서 공짜, 학습 후 헤드 제거

### 기타

- URDF
    - 실제 로봇의 URDF, USD 파일을 최대한 많이 수집
    - URDF, USD는 별도의 스크립트가 아닌, 에이전트를 통해서 직접 수집
- GT 생성을 위한 자율주행 파이프라인
    1. elevation map (Isaac Lab 자체 height scan)
    2. 목적지 - 현위지 직선 연결 (path)
    3. 경로 추종 (pure pursuit)
    4. 각 로봇 별 제어기
- 시뮬레이션
    - Isaac Sim + Lab
    - controller 학습 및 테스트 : TerrainGenerator (랜덤 경사/단차 생성)
    - Traversability GT 생성 : 현실과 유사한 씬 환경 (웹 검색 필요)
- 기타
    - Depth 증강 포함 (구멍, 가장자리 노이즈, 거리 의존 노이즈 → sim-to-real 대비)
    - 같은 이미지에 다른 URDF를 넣었을 때, 히트맵이 달라지는지 검증 필수
    - embodiment 커버리지는 수집한 로봇의 종류·대수로 결정되므로, wheeled/legged/humanoid 각각 형태·크기가 충분히 다양한 로봇을 확보하는 것이 중요

### 주장할 부분

- URDF 인코딩을 통한 Embodiment Representation for Navigation을 학습함

### 폴더 구조

```text
EA-Trav/
├── plan.md
├── docker/                     # Dockerfile, docker-compose.yml
├── data/
├── configs/                    # 지형·롤아웃·모델·학습 설정
├── src/
│   ├── models/
│   └── datasets/
├── scripts/                    # 단계별 실행 엔트리포인트
│   └── sim/                   # 지형 생성, 롤아웃, 렌더링
│        ├── env/              # 시뮬레이션 환경 (씬)
│        ├── controller/       # wheeled 제어기, RL 정책 학습·래퍼
│        └── uitls/            # isaac sim 구성을 위한 기타 도구
├── tools/                      # scripts/ 에서 작성한 프로그램의 진입점
└── check/                      # 파일럿 테스트 산출물 (+ 시각화 자료)

/workspace/ea-trav/data/           # 대용량 산출물
├── isaac-cache/
├── urdf/                       # 수집·정리된 URDF
├── sim/
│   ├── usd/                   # 수집된 USD
│   ├── policies/              # embodiment별 제어기·정책
│   └── assets/                # 지형 에셋·그리드 맵
├── datasets/
│   ├── rollouts/              # 셀별 통과 가능성 점수
│   └── train_datasets/        # 뎁스·RGB·히트맵 GT 샘플
└── checkpoints/                # 모델 학습 체크포인트
```
