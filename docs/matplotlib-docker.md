# Matplotlib backends in Docker

## How it works

Matplotlib selects a backend at import time. Inside Docker, the default is
`Agg` (non-interactive, renders to files only). To display windows you need:

1. A GUI **backend** — determines which toolkit matplotlib uses to open windows.
2. The matching **system libraries** — installed via `apt-get` (as root, before
   the `USER` switch).
3. The matching **Python bindings** — installed via `pip` inside the virtualenv.
4. A live **X11 connection** at runtime — provided by the host via socket mount.

The active backend is controlled by the environment variable:

```dockerfile
ENV MPLBACKEND=TkAgg   # or Qt5Agg, Qt6Agg, GTK3Agg, WebAgg, …
```

---

## Backend reference

### TkAgg

Simplest option. The official `python:3.x-slim` image has tkinter compiled in;
`python3-tk` only needs to supply the Tcl/Tk shared libraries at OS level — the
virtualenv does **not** need `--system-site-packages`.

```dockerfile
# apt-get
python3-tk
```

No extra pip package required.

---

### Qt5Agg

```dockerfile
# apt-get
libqt5widgets5  libqt5gui5  libqt5core5a  libqt5dbus5  libqt5x11extras5
```

```
# pip (choose one)
PyQt5
PySide2
```

---

### Qt6Agg

Package names carry a `t64` suffix on Debian Bookworm.

```dockerfile
# apt-get
libqt6widgets6t64  libqt6gui6t64  libqt6core6t64
```

```
# pip (choose one)
PyQt6
PySide6
```

---

### GTK3Agg / GTK3Cairo

```dockerfile
# apt-get
libgtk-3-0  gir1.2-gtk-3.0  python3-gi  python3-gi-cairo  libcairo2
```

```
# pip
PyGObject
pycairo
```

---

### GTK4Agg

```dockerfile
# apt-get
libgtk-4-1  gir1.2-gtk-4.0  python3-gi
```

```
# pip
PyGObject
```

---

### WebAgg

No system packages required. Matplotlib serves a browser-based canvas via
`tornado`. Useful when X11 forwarding is not available.

```
# pip (usually pulled automatically by matplotlib)
tornado
```

---

### Cairo (non-interactive file output)

```dockerfile
# apt-get
libcairo2  pkg-config
```

```
# pip (choose one)
cairocffi
pycairo
```

---

### WxAgg

Not practical on `python:*-slim` images. wxPython has no pre-built Debian
wheels and requires a full build toolchain with many GTK/X11 headers. Avoid
unless strictly necessary.

---

## Dockerfile snippet — common interactive backends

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 libxext6 libxrender1 libxcb1 \
        libxkbcommon-x11-0 libgl1 \
        python3-tk \
        libqt5widgets5 libqt5gui5 libqt5core5a libqt5dbus5 libqt5x11extras5 \
        libqt6widgets6t64 libqt6gui6t64 libqt6core6t64 \
    && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=TkAgg
```

Add `PyQt5`, `PyQt6`, or `PySide6` to your `pip install` step if you want the
Qt backends available.

---

## Runtime — X11 forwarding

The container must share the host X11 socket. Two options:

**With Xauthority (recommended):**

```sh
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$XAUTHORITY":/home/magnetrun/.Xauthority:ro \
    -e XAUTHORITY=/home/magnetrun/.Xauthority \
    magnetrun
```

**Without auth (quick test only):**

```sh
xhost +local:docker
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    magnetrun
xhost -local:docker   # restore afterwards
```
