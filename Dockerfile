# Global ARGs — must be declared before any FROM to be usable in FROM instructions
ARG CUDA_VERSION=12.6.0
ARG UBUNTU_VERSION=24.04

# ── base ──────────────────────────────────────────────────────
FROM python:3.13-slim AS base

LABEL maintainer="christophe.trophime@lncmi.cnrs.fr"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 libxext6 libxrender1 libxcb1 \
        libxkbcommon-x11-0 libgl1 python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash magnetrun
WORKDIR /home/magnetrun/app
RUN chown magnetrun:magnetrun /home/magnetrun/app
COPY --chown=magnetrun:magnetrun . .
RUN mkdir -p /mnt/LNCMIG-Data && \
    chown magnetrun:magnetrun /mnt/LNCMIG-Data

VOLUME /mnt/LNCMIG-Data

ENV VIRTUAL_ENV=/home/magnetrun/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV MPLBACKEND=TkAgg

USER magnetrun
ENV HOME=/home/magnetrun

RUN echo "alias cp='cp -i'" > $HOME/.bash_aliases && \
    echo "alias ls='ls --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias mv='mv -i'" >> $HOME/.bash_aliases && \
    echo "alias rm='rm -i'" >> $HOME/.bash_aliases

RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir -e "python_magnetcooling[all]" \
    && pip install --no-cache-dir -e ".[all]"

ENTRYPOINT ["/bin/bash"]

# ── software-gl: Mesa CPU rasteriser ─────────────────────────
FROM base AS software-gl

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-dri libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*
USER magnetrun

RUN pip install --no-cache-dir PyOpenGL PyOpenGL-accelerate

# ── egl: headless / offscreen, no X11 needed ─────────────────
FROM base AS egl

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libegl-mesa0 libgles2 \
    && rm -rf /var/lib/apt/lists/*
USER magnetrun

ENV PYOPENGL_PLATFORM=egl
RUN pip install --no-cache-dir PyOpenGL PyOpenGL-accelerate

# ── dri: Intel/AMD GPU passthrough ───────────────────────────
FROM base AS dri

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libdrm2 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
USER magnetrun

RUN pip install --no-cache-dir PyOpenGL PyOpenGL-accelerate

# ── nvidia: NVIDIA GPU (different base image) ─────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS nvidia

# deadsnakes PPA provides Python 3.13 natively compiled for Ubuntu —
# avoids cross-distro SSL/ABI issues that arise from copying the Debian build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-tk \
        libx11-6 libxext6 libxrender1 libxcb1 \
        libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash magnetrun
WORKDIR /home/magnetrun/app
RUN chown magnetrun:magnetrun /home/magnetrun/app
COPY --chown=magnetrun:magnetrun . .
RUN mkdir -p /mnt/LNCMIG-Data && \
    chown magnetrun:magnetrun /mnt/LNCMIG-Data

VOLUME /mnt/LNCMIG-Data

ENV VIRTUAL_ENV=/home/magnetrun/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV MPLBACKEND=TkAgg
ENV PYOPENGL_PLATFORM=egl

USER magnetrun
ENV HOME=/home/magnetrun

RUN python3.13 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir -e "python_magnetcooling[all]" \
    && pip install --no-cache-dir -e ".[all]" \
    && pip install --no-cache-dir PyOpenGL PyOpenGL-accelerate

ENTRYPOINT ["/bin/bash"]
