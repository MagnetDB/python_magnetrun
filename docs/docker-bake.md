# Docker Bake

Docker Bake (`docker buildx bake`) is a higher-level build tool that lets you
define multiple image targets — with shared settings, inheritance, and parallel
builds — in a single `docker-bake.hcl` file.

---

## File layout

```
.
├── Dockerfile          # multi-stage: base, webagg, software-gl, egl, dri, nvidia
└── docker-bake.hcl     # build matrix
```

---

## Dockerfile stages

The base OS is controlled by the `BASE_IMAGE` build argument (default:
`debian:trixie-slim`). Each non-nvidia stage inherits from `base`; `nvidia`
redefines from a CUDA image.

| Stage | Base | Extra packages / pip | Use case |
|---|---|---|---|
| `base` | `${BASE_IMAGE}` | X11+xcb, python3-tk, PyQt5, PyQt6 | common setup |
| `webagg` | `base` | `tornado` | browser-based, no display needed |
| `software-gl` | `base` | `libgl1-mesa-dri`, `libglu1-mesa`, PyOpenGL | CPU OpenGL + X11 |
| `egl` | `base` | `libegl1`, `libegl-mesa0`, `libgles2`, PyOpenGL | headless / offscreen |
| `dri` | `base` | `libdrm2`, `libgl1-mesa-dri`, PyOpenGL | Intel/AMD GPU |
| `nvidia` | `nvidia/cuda:X-runtime-ubuntuY` | full setup replicated, PyOpenGL | NVIDIA GPU + CUDA |

See [docker-opengl.md](docker-opengl.md) for OpenGL package details per scenario.

### Matplotlib backend support

All display stages (`base`, `software-gl`, `egl`, `dri`, `nvidia`) ship with
**TkAgg, Qt5Agg and Qt6Agg** pre-installed. The active backend defaults to
`TkAgg` and is switched at runtime via `MPLBACKEND`:

```sh
docker run -e MPLBACKEND=Qt6Agg … magnetrun:software-gl-trixie-latest
```

The `webagg` stage pre-sets `MPLBACKEND=WebAgg` and exposes port `8988`.

---

## Build matrix

Targets are named `{stage}-{os}`. Supported OS values:

| Tag suffix | Base image |
|---|---|
| `debian13` | `debian:13-slim` |
| `trixie` | `debian:trixie-slim` |
| `forky` | `debian:forky-slim` *(Debian 14 — image may not be published yet)* |
| `ubuntu24.04` | `ubuntu:24.04` |
| `ubuntu26.04` | `ubuntu:26.04` |

`nvidia` targets exist only for Ubuntu (`nvidia-ubuntu24.04`, `nvidia-ubuntu26.04`).

### Groups

| Group | Targets |
|---|---|
| `default` | software-gl/egl/dri/webagg on `debian:trixie-slim` |
| `debian-13` | software-gl/egl/dri/webagg on `debian:13-slim` |
| `debian-trixie` | software-gl/egl/dri/webagg on `debian:trixie-slim` |
| `debian-forky` | software-gl/egl/dri/webagg on `debian:forky-slim` |
| `ubuntu-2404` | software-gl/egl/dri/webagg + nvidia on `ubuntu:24.04` |
| `ubuntu-2604` | software-gl/egl/dri/webagg + nvidia on `ubuntu:26.04` |
| `webagg` | webagg on trixie, ubuntu:24.04, ubuntu:26.04 |
| `all` | all targets except forky and debian-13 |

---

## Common commands

### Build the default group (trixie, all GL modes + webagg)

```sh
docker buildx bake
```

### Build all targets for a specific OS

```sh
docker buildx bake debian-trixie
docker buildx bake ubuntu-2404
docker buildx bake ubuntu-2604
```

### Build all webagg images

```sh
docker buildx bake webagg
```

### Build all targets

```sh
docker buildx bake all
```

### Build a single target

```sh
docker buildx bake egl-trixie
docker buildx bake webagg-ubuntu2404
```

### Override the image tag

```sh
TAG=v1.2 docker buildx bake debian-trixie
```

### Override the base image at build time

```sh
docker buildx bake egl-trixie \
  --set egl-trixie.args.BASE_IMAGE=debian:trixie-slim
```

### Dry-run — print what would be built without building

```sh
docker buildx bake --print
docker buildx bake ubuntu-2404 --print
```

### Push to a registry

```sh
TAG=v1.2 docker buildx bake --push
```

---

## Running each image

### Mounting LNCMIG-Data from the host

All images expose `/mnt/LNCMIG-Data` as a `VOLUME`. Pass `-v` to bind-mount
from any host path:

```sh
-v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data    # host path is /mnt/LNCMIG-Data
-v "$HOME/LNCMIG-Data":/mnt/LNCMIG-Data  # host path is ~/LNCMIG-Data
```

---

### software-gl — X11 window, CPU (Mesa) rendering

```sh
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:software-gl-trixie-latest
```

Switch to Qt5Agg or Qt6Agg at runtime:

```sh
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -e MPLBACKEND=Qt6Agg \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:software-gl-trixie-latest
```

### egl — headless OpenGL + matplotlib via X11

`PYOPENGL_PLATFORM=egl` only affects PyOpenGL-based rendering. Matplotlib uses
the `MPLBACKEND` backend, which still needs X11 for display (TkAgg/Qt backends).

```sh
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:egl-trixie-latest
```

For truly headless use (save figures to files only), omit the display flags and
use the non-interactive `Agg` backend:

```sh
docker run -it --rm \
    -e MPLBACKEND=Agg \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:egl-trixie-latest
```

### dri — Intel/AMD GPU + X11

```sh
docker run -it --rm \
    --device /dev/dri:/dev/dri \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:dri-trixie-latest
```

### webagg — browser-based, no display server needed

Start the container with port `8988` exposed, then open `http://localhost:8988`
in any browser on the host:

```sh
docker run -it --rm \
    -p 8988:8988 \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:webagg-trixie-latest
```

No X11 socket or `DISPLAY` variable is needed. Useful for remote / headless
servers where opening a GUI window is not possible.

### nvidia — NVIDIA GPU (requires NVIDIA Container Toolkit on host)

```sh
docker run -it --rm \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/LNCMIG-Data:/mnt/LNCMIG-Data \
    magnetrun:nvidia-ubuntu24.04-latest
```

---

### Shell helper

Define a shell function in `~/.bashrc` or `~/.zshrc` to avoid repeating flags.
Set `MAGNETRUN_IMAGE` to pick the image and `MPLBACKEND` to pick the backend:

```sh
magnetrun() {
  local image="${MAGNETRUN_IMAGE:-magnetrun:software-gl-trixie-latest}"

  local data_host
  if [ -d /mnt/LNCMIG-Data ]; then
    data_host=/mnt/LNCMIG-Data
  else
    data_host="$HOME/LNCMIG-Data"
  fi

  docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -e MPLBACKEND="${MPLBACKEND:-TkAgg}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$data_host":/mnt/LNCMIG-Data \
    "$image" "$@"
}
```

Usage:

```sh
magnetrun                                                    # interactive shell, TkAgg
MPLBACKEND=Qt6Agg magnetrun                                  # switch to Qt6
MAGNETRUN_IMAGE=magnetrun:webagg-trixie-latest magnetrun     # use webagg image
magnetrun -c "magnetrun plot run.txt"                        # run a command
```

---

## Bake file tips

### Caching with a registry

```hcl
target "egl-trixie" {
  inherits   = ["_common"]
  target     = "egl"
  args       = { BASE_IMAGE = "debian:trixie-slim" }
  tags       = ["magnetrun:egl-trixie-${TAG}"]
  cache-from = ["type=registry,ref=magnetrun:egl-trixie-cache"]
  cache-to   = ["type=registry,ref=magnetrun:egl-trixie-cache,mode=max"]
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

### Adding a new OS target

To add, say, `ubuntu:25.10`:

1. Add targets to `docker-bake.hcl`:

```hcl
target "software-gl-ubuntu2510" {
  inherits = ["_common"]
  target   = "software-gl"
  args     = { BASE_IMAGE = "ubuntu:25.10" }
  tags     = ["magnetrun:software-gl-ubuntu25.10-${TAG}"]
}
# … repeat for egl, dri, webagg
```

2. Add a group:

```hcl
group "ubuntu-2510" {
  targets = ["software-gl-ubuntu2510", "egl-ubuntu2510", "dri-ubuntu2510", "webagg-ubuntu2510"]
}
```

No Dockerfile changes are needed; the OS default `python3` is used automatically.

---

## Prerequisites

- Docker Engine ≥ 23 (Bake is stable from this version)
- `docker buildx` plugin (included by default since Docker Desktop 4.x and
  `docker-ce` on Linux with the `docker-buildx-plugin` package)

```sh
# verify
docker buildx version
```
