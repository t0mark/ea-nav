FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git wget curl unzip \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    libvulkan1 mesa-vulkan-drivers \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Core ML
# numpy<2.0.0: MANSION/objathor 생태계가 numpy 1.x를 요구함(아래에서 다시 고정).
RUN pip install --no-cache-dir \
    transformers \
    ftfy regex tqdm \
    "numpy<2.0.0" \
    opencv-python-headless \
    Pillow imageio scikit-image

# CLIP
RUN pip install --no-cache-dir \
    "git+https://github.com/openai/CLIP.git"

# Graph / planning
RUN pip install --no-cache-dir \
    networkx scipy

# API clients (검증 에이전트용)
RUN pip install --no-cache-dir \
    anthropic openai

# Data tools
RUN pip install --no-cache-dir \
    hydra-core omegaconf pyyaml h5py

# Qwen2.5-VL 유틸리티 (VLM 명령 생성용)
RUN pip install --no-cache-dir \
    qwen-vl-utils \
    accelerate

# ── MANSION (airlab_hw_mansion 서비스 전용) ─────────────────────────────────
# https://github.com/AgibotGeneral/MANSION (Apache-2.0, Holodeck 기반)
# 멀티플로어 빌딩 + 실제 작동하는 Stairs/Elevator 자산.

# 1) MANSION이 명시하는 Python 의존성(생성 파이프라인 + LLM 클라이언트 포함).
#    pydantic 상한(<2.10.0)은 MANSION 자체 요구사항.
RUN pip install --no-cache-dir \
    sentence-transformers \
    langchain-openai langgraph \
    procthor objathor objaverse \
    shapely rtree trimesh cvxpy \
    "pillow<11.0.0" moviepy \
    loguru compress-json compress-pickle open-clip-torch psutil \
    python-dotenv "pydantic>=2.0.0,<2.10.0" importlib-metadata editdistance \
    "flask<3.1.0" "werkzeug<3.1.0" \
    attrs colorama

# 2) MANSION이 고정하는 패치 버전 AI2-THOR (일반 pip ai2thor 대신 이 커밋을 사용해야
#    MANSION의 Stairs/Elevator/door 자산과 호환됨).
RUN pip install --no-cache-dir \
    --extra-index-url https://ai2thor-pypi.allenai.org \
    ai2thor==0+6f165fdaf3cf2d03728f931f39261d14a67414d0

# 3) Mansion Gym(mansion-api): MansionWorld 데이터셋을 로드하는 gymnasium 환경.
RUN pip install --no-cache-dir \
    gymnasium termcolor mansion-api

# 4) MANSION 레포(생성 파이프라인 + setup_mansion.py 패치 설치 스크립트).
#    objathor 자산/mansion_patch.zip 다운로드는 용량이 크고(~수GB) 실패 가능성이
#    있어 이미지 빌드에 굽지 않고 컨테이너 런타임(첫 실행)에서 받는다.
RUN git clone --depth 1 https://github.com/AgibotGeneral/MANSION.git /opt/mansion

ENV MANSION_HOME=/opt/mansion
ENV OBJATHOR_ASSETS_VERSION=2023_09_23

# ── GPU 렌더링 (VirtualGL EGL) ──────────────────────────────────────────────
# MANSION의 AI2-THOR Unity 빌드는 OpenGL 전용(Vulkan 미포함)이라 CloudRendering
# 불가. Xvfb만으로는 llvmpipe 소프트웨어 렌더링으로 떨어지므로, VirtualGL의
# EGL 백엔드로 GL 호출을 NVIDIA GPU에 오프로드한다.
# 사용: docker exec airlab_hw_mansion vglrun -d egl0 python <스크립트>  (egl0~3 = GPU 0~3)
RUN apt-get update && apt-get install -y vulkan-tools mesa-utils \
    && wget -q https://github.com/VirtualGL/virtualgl/releases/download/3.1.1/virtualgl_3.1.1_amd64.deb -O /tmp/vgl.deb \
    && apt-get install -y /tmp/vgl.deb && rm /tmp/vgl.deb \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/VirtualGL/bin:$PATH

# ── URDF 랜덤 생성 파이프라인 (scripts/datasets/URDF/) ──────────────────────
# pybullet: 정적 검증(자기충돌·AABB·지지다각형) + TinyRenderer 파일럿 카드 렌더.
# yourdfpy: URDF 파싱 왕복 검증. (컨테이너 내부 설치 검증 완료 후 반영, 2026-07-25)
RUN pip install --no-cache-dir pybullet yourdfpy

WORKDIR /workspace
