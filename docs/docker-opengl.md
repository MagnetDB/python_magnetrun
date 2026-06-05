# OpenGL and EGL in Docker

## Concepts

| Term | Role |
|---|---|
| **OpenGL** | Rendering API — draws geometry, textures, shaders |
| **EGL** | Platform interface — connects OpenGL/ES to the display or offscreen surface |
| **Mesa** | Open-source OpenGL/EGL implementation; provides both software (CPU) and hardware (DRI) drivers |
| **DRI** | Direct Rendering Infrastructure — kernel interface for GPU access |

---

## Scenario 1 — Software rendering (CPU, Mesa llvmpipe)

No GPU required. Mesa runs the full OpenGL pipeline on CPU via the `llvmpipe`
software rasteriser. Requires an X11 connection for window display.

### System packages

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-dri \
        libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*
```

### Python packages (pip)

```
PyOpenGL
PyOpenGL-accelerate
```

### Runtime

```sh
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    magnetrun:software-gl
```

---

## Scenario 2 — Headless EGL (offscreen, no display required)

EGL can open an offscreen surface (`EGL_PLATFORM=surfaceless`) without a
running X server. Useful on headless servers, CI pipelines, or render farms.
Mesa provides a software EGL driver; an NVIDIA or DRI device can replace it
for hardware acceleration.

### System packages

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 \
        libegl-mesa0 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*
```

### Environment variable

```dockerfile
ENV PYOPENGL_PLATFORM=egl
```

This tells PyOpenGL to initialise via EGL instead of GLX (the X11 path).

### Python packages (pip)

```
PyOpenGL
PyOpenGL-accelerate
```

### Runtime — no X11 needed

```sh
docker run -it --rm magnetrun:egl
```

---

## Scenario 3 — Intel / AMD GPU (DRI passthrough)

The host kernel DRI device (`/dev/dri/renderD128`) is passed into the
container. Mesa picks up the hardware driver automatically.

### System packages

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libdrm2 \
        libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
```

### Runtime

```sh
docker run -it --rm \
    --device /dev/dri:/dev/dri \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    magnetrun:dri
```

For headless EGL with GPU acceleration, add `ENV PYOPENGL_PLATFORM=egl` and
drop the display flags.

---

## Scenario 4 — NVIDIA GPU

NVIDIA requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
on the host. Two sub-cases:

### 4a — Runtime driver mount only (no CUDA in image)

Keep the standard `python:3.13-slim` base image; the toolkit mounts the NVIDIA
libraries at `docker run` time. Sufficient for PyOpenGL, vispy, PyVista etc.
that do not need CUDA compiled extensions.

```sh
docker run -it --rm --gpus all \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    magnetrun:egl
```

### 4b — CUDA inside the image

Switch the base image to include CUDA runtime libraries:

```dockerfile
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS nvidia
```

Then install Python 3.13, recreate the user and venv setup (the `nvidia/cuda`
base image is Ubuntu, not Debian slim, so Python 3.13 must be installed
explicitly). See [docker-bake.md](docker-bake.md) for the full stage definition.

```sh
docker run -it --rm --gpus all magnetrun:nvidia
```

---

## Choosing the right scenario

| Need | Scenario |
|---|---|
| Local desktop with X11, no GPU | software-gl |
| Headless server / CI | egl |
| Local desktop with Intel/AMD GPU | dri |
| NVIDIA GPU, no CUDA extensions needed | egl image + `--gpus all` |
| NVIDIA GPU + CUDA compiled extensions | nvidia image + `--gpus all` |

---

## Verifying OpenGL inside the container

```sh
# software or DRI
glxinfo | grep "OpenGL renderer"

# EGL / headless
python -c "
import OpenGL.EGL as egl
d = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
egl.eglInitialize(d, None, None)
print('EGL OK')
"

# PyOpenGL version
python -c "import OpenGL; print(OpenGL.__version__)"
```
