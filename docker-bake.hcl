variable "TAG" {
  default = "latest"
}

variable "CUDA_VERSION" {
  default = "12.6.0"
}

# default group — excludes nvidia (requires host toolkit)
group "default" {
  targets = ["software-gl", "egl", "dri"]
}

group "all" {
  targets = ["software-gl", "egl", "dri", "nvidia-2404"]
}

# nvidia-2604 kept for future use — uncomment in "all" once
# docker.io/nvidia/cuda:X.Y.Z-runtime-ubuntu26.04 is published
group "nvidia-all" {
  targets = ["nvidia-2404", "nvidia-2604"]
}

# ── shared settings ──────────────────────────────────────────
target "_common" {
  dockerfile = "Dockerfile"
  context    = "."
  output     = ["type=docker"]
}

# ── targets ──────────────────────────────────────────────────
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

target "nvidia-2404" {
  inherits = ["_common"]
  target   = "nvidia"
  args     = {
    CUDA_VERSION   = CUDA_VERSION
    UBUNTU_VERSION = "24.04"
  }
  tags = ["magnetrun:nvidia-ubuntu24.04-${TAG}"]
}

target "nvidia-2604" {
  inherits = ["_common"]
  target   = "nvidia"
  args     = {
    CUDA_VERSION   = CUDA_VERSION
    UBUNTU_VERSION = "26.04"
  }
  tags = ["magnetrun:nvidia-ubuntu26.04-${TAG}"]
}
