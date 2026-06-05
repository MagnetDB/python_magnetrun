# Docker Bake

Docker Bake (`docker buildx bake`) is a higher-level build tool that lets you
define multiple image targets — with shared settings, inheritance, and parallel
builds — in a single `docker-bake.hcl` file.

---

## File layout

```
.
├── Dockerfile          # multi-stage: base, software-gl, egl, dri, nvidia
└── docker-bake.hcl     # build matrix
```

---

## Dockerfile stages

Each stage extends `base` (or redefines from scratch for `nvidia`):

| Stage | Base image | Extra packages | Use case |
|---|---|---|---|
| `base` | `python:3.13-slim` | X11, TkAgg | common setup |
| `software-gl` | `base` | `libgl1-mesa-dri`, `libglu1-mesa` | CPU OpenGL + X11 |
| `egl` | `base` | `libegl1`, `libegl-mesa0`, `libgles2` | headless, no display |
| `dri` | `base` | `libdrm2`, `libgl1-mesa-dri` | Intel/AMD GPU |
| `nvidia` | `nvidia/cuda:12.6.0-runtime-ubuntu24.04` | full setup replicated | NVIDIA GPU + CUDA |

See [docker-opengl.md](docker-opengl.md) for the apt and pip package details
per scenario.

---

## docker-bake.hcl

```hcl
variable "TAG" {
  default = "latest"
}

# default group — excludes nvidia (requires host toolkit)
group "default" {
  targets = ["software-gl", "egl", "dri"]
}

group "all" {
  targets = ["software-gl", "egl", "dri", "nvidia"]
}

# shared settings inherited by all targets
target "_common" {
  dockerfile = "Dockerfile"
  context    = "."
}

target "software-gl" {
  inherits = ["_common"]
  target   = "software-gl"
  tags     = ["magnetrun:software-gl-${TAG}"]
}

target "egl" {
  inherits = ["_common"]
  target   = "egl"
  tags     = ["magnetrun:egl-${TAG}"]
}

target "dri" {
  inherits = ["_common"]
  target   = "dri"
  tags     = ["magnetrun:dri-${TAG}"]
}

target "nvidia" {
  inherits = ["_common"]
  target   = "nvidia"
  tags     = ["magnetrun:nvidia-${TAG}"]
}
```

---

## Common commands

### Build the default group (software-gl, egl, dri) in parallel

```sh
docker buildx bake
```

### Build all targets including nvidia

```sh
docker buildx bake all
```

### Build a single target

```sh
docker buildx bake egl
```

### Override the image tag

```sh
TAG=v1.2 docker buildx bake
```

### Dry-run — print what would be built without building

```sh
docker buildx bake --print
```

### Push to a registry

```sh
TAG=v1.2 docker buildx bake --push
```

---

## Running each image

### Mounting LNCMIG-Data from the host

The images expose `/mnt/LNCMIG-Data` as a `VOLUME`. Pass `-v` to bind-mount
the host directory at that path regardless of where it lives on the host:

```sh
# host path is /mnt/LNCMIG-Data
-v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data

# host path is ~/LNCMIG-Data
-v "$HOME/LNCMIG-Data":/mnt/LNCMIG-Data
```

Both forms map to the same container path so application code always reads
from `/mnt/LNCMIG-Data` regardless of the host layout.

---

### software-gl — X11 window, CPU (Mesa) rendering

```sh
# with /mnt/LNCMIG-Data on host
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:software-gl-latest

# with ~/LNCMIG-Data on host
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/LNCMIG-Data":/mnt/LNCMIG-Data \
    magnetrun:software-gl-latest
```

### egl — OpenGL headless + matplotlib via X11/Tk

`PYOPENGL_PLATFORM=egl` only affects PyOpenGL-based rendering. Matplotlib uses
the TkAgg backend, which still needs X11 to open windows.

```sh
# with /mnt/LNCMIG-Data on host
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:egl-latest

# with ~/LNCMIG-Data on host
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/LNCMIG-Data":/mnt/LNCMIG-Data \
    magnetrun:egl-latest
```

For truly headless use (no display, save figures to files only), omit the
`-e DISPLAY` and `-v /tmp/.X11-unix` flags and set `MPLBACKEND=Agg` at
runtime: `-e MPLBACKEND=Agg`.

### dri — Intel/AMD GPU + X11

```sh
# with /mnt/LNCMIG-Data on host
docker run -it --rm \
    --device /dev/dri:/dev/dri \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:dri-latest

# with ~/LNCMIG-Data on host
docker run -it --rm \
    --device /dev/dri:/dev/dri \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/LNCMIG-Data":/mnt/LNCMIG-Data \
    magnetrun:dri-latest
```

### nvidia — NVIDIA GPU (requires NVIDIA Container Toolkit on host)

```sh
# with /mnt/LNCMIG-Data on host
docker run -it --rm \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:nvidia-ubuntu24.04-latest

# with ~/LNCMIG-Data on host
docker run -it --rm \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/LNCMIG-Data":/mnt/LNCMIG-Data \
    magnetrun:nvidia-ubuntu24.04-latest
```

---

### Shell helper

To avoid repeating flags, define a shell function in `~/.bashrc` or `~/.zshrc`:

```sh
magnetrun() {
  local image="${MAGNETRUN_IMAGE:-magnetrun:egl-latest}"

  # resolve data directory: prefer /mnt/LNCMIG-Data, fall back to ~/LNCMIG-Data
  local data_host
  if [ -d /mnt/LNCMIG-Data ]; then
    data_host=/mnt/LNCMIG-Data
  else
    data_host="$HOME/LNCMIG-Data"
  fi

  docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$data_host":/mnt/LNCMIG-Data \
    "$image" "$@"
}
```

Usage:

```sh
magnetrun                                       # interactive shell
magnetrun -c "magnetrun plot run.txt"           # run a command
MAGNETRUN_IMAGE=magnetrun:dri-latest magnetrun  # use a different image
```

---

## Bake file tips

### Caching with a registry

```hcl
target "egl" {
  inherits   = ["_common"]
  target     = "egl"
  tags       = ["magnetrun:egl-${TAG}"]
  cache-from = ["type=registry,ref=magnetrun:egl-cache"]
  cache-to   = ["type=registry,ref=magnetrun:egl-cache,mode=max"]
}
```

### Multi-platform builds

```hcl
target "_common" {
  dockerfile = "Dockerfile"
  context    = "."
  platforms  = ["linux/amd64", "linux/arm64"]
}
```

Requires a multi-platform builder:
```sh
docker buildx create --use --name multi
```

### Build-time variables via args

```hcl
variable "CUDA_VERSION" {
  default = "12.6.0"
}

target "nvidia" {
  inherits = ["_common"]
  target   = "nvidia"
  args     = { CUDA_VERSION = CUDA_VERSION }
  tags     = ["magnetrun:nvidia-${TAG}"]
}
```

Then in the Dockerfile:
```dockerfile
ARG CUDA_VERSION=12.6.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS nvidia
```

---

## Prerequisites

- Docker Engine ≥ 23 (Bake is stable from this version)
- `docker buildx` plugin (included by default since Docker Desktop 4.x and
  `docker-ce` on Linux with the `docker-buildx-plugin` package)

```sh
# verify
docker buildx version
```
