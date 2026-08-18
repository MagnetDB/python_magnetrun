---
mode: agent
description: Set up Dagger to run CI pipelines locally for a Python project, mirroring the GitHub Actions workflows.
---

# Set up Dagger for local CI testing

Set up [Dagger](https://dagger.io) in this Python project so that the GitHub Actions CI workflows can be run locally.

## Context

This project uses GitHub Actions with two workflows:
- **test.yml** — runs `pytest` across multiple Python versions (3.11–3.14) on Ubuntu, and also on a Debian Trixie container
- **docs.yml** — builds Sphinx documentation with `pip install -e ".[doc]"` then `cd docs && make html`

Dependencies are managed via `pyproject.toml` with extras: `[dev]` for tests, `[doc]` for docs.

## Steps to perform

### 1. Install the Dagger CLI (if not already installed)

Run in terminal:
```bash
curl -fsSL https://dl.dagger.io/dagger/install.sh | BIN_DIR=/usr/local/bin sudo -E sh
dagger version
```

Ensure Docker (or Podman) is running — Dagger requires a container runtime.

### 2. Initialize a Dagger Python module

In the project root:
```bash
dagger init --sdk=python --name=ci
```

This creates `.dagger/` with `main.py` and `pyproject.toml`.

### 3. Implement the CI functions

Create `.dagger/src/ci/main.py` with the following Dagger functions mirroring the GitHub Actions jobs:

- **`test(source, python_version)`** — installs `.[dev]` extras and runs `pytest -v --tb=short` inside the given Python container image
- **`test_matrix(source)`** — calls `test()` for all versions: `3.11`, `3.12`, `3.13`, `3.14`
- **`test_debian(source)`** — runs tests inside `debian:trixie`, using a venv with `--system-site-packages`
- **`build_docs(source)`** — installs `.[doc]` extras, runs `cd docs && make html`, and exports the built HTML

All functions should accept a `source: dagger.Directory` argument pointing to the project root.

### 4. Run CI locally

```bash
# Run tests for a specific Python version
dagger call test --source=. --python-version=3.12

# Run the full test matrix
dagger call test-matrix --source=.

# Run Debian Trixie tests
dagger call test-debian --source=.

# Build documentation
dagger call build-docs --source=.
```

### 5. (Optional) Dagger Cloud tracing

Sign up for free at https://dagger.cloud to get a browser-based trace UI for every run:
```bash
dagger login
```

## Notes

- The `.dagger/` directory should be committed to the repository.
- Cache volumes can be added for `pip` to speed up repeated runs.
- The `source` directory argument is how Dagger accesses host files — it is sandboxed by design.
- Function names use kebab-case on the CLI (`test-matrix`) but snake_case in Python (`test_matrix`).
