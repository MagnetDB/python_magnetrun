# Global ARGs — must be declared before any FROM to be usable in FROM instructions
ARG BASE_IMAGE=debian:trixie-slim
ARG CUDA_VERSION=12.6.0
ARG UBUNTU_VERSION=24.04

# ── base ──────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS base

LABEL maintainer="christophe.trophime@lncmi.cnrs.fr"

# xcb plugin libs are required by pip-installed PyQt5/PyQt6 wheels (which
# bundle Qt itself but depend on the system xcb platform plugin chain).
# This single set of packages enables TkAgg, Qt5Agg and Qt6Agg; the active
# backend is selected at runtime via MPLBACKEND.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python-is-python3 \
        python3-venv \
        python3-tk \
        libx11-6 libxext6 libxrender1 \
        libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 \
        libxkbcommon-x11-0 libgl1 \
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
# Default display backend; override at runtime: docker run -e MPLBACKEND=Qt6Agg …
ENV MPLBACKEND=TkAgg

USER magnetrun
ENV HOME=/home/magnetrun

RUN echo "alias cp='cp -i'" > $HOME/.bash_aliases && \
    echo "alias ls='ls --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias mv='mv -i'" >> $HOME/.bash_aliases && \
    echo "alias rm='rm -i'" >> $HOME/.bash_aliases

RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir -e "python_magnetcooling[all]" \
    && pip install --no-cache-dir -e ".[all]" \
    && pip install --no-cache-dir PyQt5 PyQt6

ENTRYPOINT ["/bin/bash"]

# ── webagg: browser-based, no display server needed ──────────
# Access figures at http://localhost:8988 after starting the app.
FROM base AS webagg
RUN pip install --no-cache-dir tornado
ENV MPLBACKEND=WebAgg
EXPOSE 8988

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

# ── nvidia: NVIDIA GPU (nvidia/cuda base, ubuntu only) ────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS nvidia

RUN apt-get update && apt-get install -y --no-install-recommends \
        python-is-python3 \
        python3-venv \
        python3-tk \
        libx11-6 libxext6 libxrender1 \
        libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 \
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

RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --no-cache-dir -e "python_magnetcooling[all]" \
    && pip install --no-cache-dir -e ".[all]" \
    && pip install --no-cache-dir PyQt5 PyQt6 PyOpenGL PyOpenGL-accelerate

ENTRYPOINT ["/bin/bash"]
