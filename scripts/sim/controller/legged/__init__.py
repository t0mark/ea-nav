"""legged 제어기 패키지: RL 정책 저수준 (low_rl) + 조립 (controller).

학습 인프라는 train/ 서브패키지에 격리 — 배포 경로(controller)는 train/을
임포트하지 않는다 (rsl-rl·학습 의존성이 3단계 롤아웃에 새지 않게).
"""
