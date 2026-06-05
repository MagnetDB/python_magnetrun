variable "TAG" {
  default = "latest"
}

variable "CUDA_VERSION" {
  default = "12.6.0"
}

# ── groups ────────────────────────────────────────────────────

# default: trixie, all GL modes + webagg (no nvidia)
group "default" {
  targets = ["software-gl-trixie", "egl-trixie", "dri-trixie", "webagg-trixie"]
}

group "debian-13" {
  targets = ["software-gl-debian13", "egl-debian13", "dri-debian13", "webagg-debian13"]
}

group "debian-trixie" {
  targets = ["software-gl-trixie", "egl-trixie", "dri-trixie", "webagg-trixie"]
}

# debian:forky-slim (Debian 14 — add to builds once image is published)
group "debian-forky" {
  targets = ["software-gl-forky", "egl-forky", "dri-forky", "webagg-forky"]
}

group "ubuntu-2404" {
  targets = [
    "software-gl-ubuntu2404", "egl-ubuntu2404", "dri-ubuntu2404",
    "webagg-ubuntu2404", "nvidia-ubuntu2404",
  ]
}

group "ubuntu-2604" {
  targets = [
    "software-gl-ubuntu2604", "egl-ubuntu2604", "dri-ubuntu2604",
    "webagg-ubuntu2604", "nvidia-ubuntu2604",
  ]
}

# webagg only — useful for headless / server deployments
group "webagg" {
  targets = ["webagg-trixie", "webagg-ubuntu2404", "webagg-ubuntu2604"]
}

group "all" {
  targets = [
    "software-gl-trixie", "egl-trixie", "dri-trixie", "webagg-trixie",
    "software-gl-ubuntu2404", "egl-ubuntu2404", "dri-ubuntu2404",
    "webagg-ubuntu2404", "nvidia-ubuntu2404",
    "software-gl-ubuntu2604", "egl-ubuntu2604", "dri-ubuntu2604",
    "webagg-ubuntu2604", "nvidia-ubuntu2604",
  ]
}

# ── shared settings ───────────────────────────────────────────
target "_common" {
  dockerfile = "Dockerfile"
  context    = "."
  output     = ["type=docker"]
}

# ── debian:13 targets ─────────────────────────────────────────
target "software-gl-debian13" {
  inherits = ["_common"]
  target   = "software-gl"
  args     = { BASE_IMAGE = "debian:13-slim" }
  tags     = ["magnetrun:software-gl-debian13-${TAG}"]
}

target "egl-debian13" {
  inherits = ["_common"]
  target   = "egl"
  args     = { BASE_IMAGE = "debian:13-slim" }
  tags     = ["magnetrun:egl-debian13-${TAG}"]
}

target "dri-debian13" {
  inherits = ["_common"]
  target   = "dri"
  args     = { BASE_IMAGE = "debian:13-slim" }
  tags     = ["magnetrun:dri-debian13-${TAG}"]
}

target "webagg-debian13" {
  inherits = ["_common"]
  target   = "webagg"
  args     = { BASE_IMAGE = "debian:13-slim" }
  tags     = ["magnetrun:webagg-debian13-${TAG}"]
}

# ── debian:trixie targets (same image as debian:13) ──────────
target "software-gl-trixie" {
  inherits = ["_common"]
  target   = "software-gl"
  args     = { BASE_IMAGE = "debian:trixie-slim" }
  tags     = ["magnetrun:software-gl-trixie-${TAG}"]
}

target "egl-trixie" {
  inherits = ["_common"]
  target   = "egl"
  args     = { BASE_IMAGE = "debian:trixie-slim" }
  tags     = ["magnetrun:egl-trixie-${TAG}"]
}

target "dri-trixie" {
  inherits = ["_common"]
  target   = "dri"
  args     = { BASE_IMAGE = "debian:trixie-slim" }
  tags     = ["magnetrun:dri-trixie-${TAG}"]
}

target "webagg-trixie" {
  inherits = ["_common"]
  target   = "webagg"
  args     = { BASE_IMAGE = "debian:trixie-slim" }
  tags     = ["magnetrun:webagg-trixie-${TAG}"]
}

# ── debian:forky targets (Debian 14 — image may not yet exist)
target "software-gl-forky" {
  inherits = ["_common"]
  target   = "software-gl"
  args     = { BASE_IMAGE = "debian:forky-slim" }
  tags     = ["magnetrun:software-gl-forky-${TAG}"]
}

target "egl-forky" {
  inherits = ["_common"]
  target   = "egl"
  args     = { BASE_IMAGE = "debian:forky-slim" }
  tags     = ["magnetrun:egl-forky-${TAG}"]
}

target "dri-forky" {
  inherits = ["_common"]
  target   = "dri"
  args     = { BASE_IMAGE = "debian:forky-slim" }
  tags     = ["magnetrun:dri-forky-${TAG}"]
}

target "webagg-forky" {
  inherits = ["_common"]
  target   = "webagg"
  args     = { BASE_IMAGE = "debian:forky-slim" }
  tags     = ["magnetrun:webagg-forky-${TAG}"]
}

# ── ubuntu:24.04 targets ──────────────────────────────────────
target "software-gl-ubuntu2404" {
  inherits = ["_common"]
  target   = "software-gl"
  args     = { BASE_IMAGE = "ubuntu:24.04" }
  tags     = ["magnetrun:software-gl-ubuntu24.04-${TAG}"]
}

target "egl-ubuntu2404" {
  inherits = ["_common"]
  target   = "egl"
  args     = { BASE_IMAGE = "ubuntu:24.04" }
  tags     = ["magnetrun:egl-ubuntu24.04-${TAG}"]
}

target "dri-ubuntu2404" {
  inherits = ["_common"]
  target   = "dri"
  args     = { BASE_IMAGE = "ubuntu:24.04" }
  tags     = ["magnetrun:dri-ubuntu24.04-${TAG}"]
}

target "webagg-ubuntu2404" {
  inherits = ["_common"]
  target   = "webagg"
  args     = { BASE_IMAGE = "ubuntu:24.04" }
  tags     = ["magnetrun:webagg-ubuntu24.04-${TAG}"]
}

target "nvidia-ubuntu2404" {
  inherits = ["_common"]
  target   = "nvidia"
  args     = {
    CUDA_VERSION   = CUDA_VERSION
    UBUNTU_VERSION = "24.04"
  }
  tags = ["magnetrun:nvidia-ubuntu24.04-${TAG}"]
}

# ── ubuntu:26.04 targets ──────────────────────────────────────
target "software-gl-ubuntu2604" {
  inherits = ["_common"]
  target   = "software-gl"
  args     = { BASE_IMAGE = "ubuntu:26.04" }
  tags     = ["magnetrun:software-gl-ubuntu26.04-${TAG}"]
}

target "egl-ubuntu2604" {
  inherits = ["_common"]
  target   = "egl"
  args     = { BASE_IMAGE = "ubuntu:26.04" }
  tags     = ["magnetrun:egl-ubuntu26.04-${TAG}"]
}

target "dri-ubuntu2604" {
  inherits = ["_common"]
  target   = "dri"
  args     = { BASE_IMAGE = "ubuntu:26.04" }
  tags     = ["magnetrun:dri-ubuntu26.04-${TAG}"]
}

target "webagg-ubuntu2604" {
  inherits = ["_common"]
  target   = "webagg"
  args     = { BASE_IMAGE = "ubuntu:26.04" }
  tags     = ["magnetrun:webagg-ubuntu26.04-${TAG}"]
}

target "nvidia-ubuntu2604" {
  inherits = ["_common"]
  target   = "nvidia"
  args     = {
    CUDA_VERSION   = CUDA_VERSION
    UBUNTU_VERSION = "26.04"
  }
  tags = ["magnetrun:nvidia-ubuntu26.04-${TAG}"]
}
