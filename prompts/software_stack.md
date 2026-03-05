# Software Stack Recommendations

This document consolidates every library, tool, and dependency discussed across the
five implementation phases. The **Mandatory in** column indicates the earliest phase
where the package becomes a hard or soft requirement; packages introduced as optional
extras may only become mandatory in a later phase.

---

## Quick-reference table

| Package | Version | `pyproject.toml` group | Mandatory in | Replaces / Notes |
|---------|---------|------------------------|--------------|-----------------|
| **ruff** | `>=0.4` | `[dev]` | Phase 1 | Replaces `flake8` + `isort` + `pyupgrade` |
| **pytest** | `>=8.0` | `[dev]` | Phase 1 | Already used; must be in `[dev]` extras |
| **pytest-cov** | `>=5.0` | `[dev]` | Phase 1 | Coverage reporting in CI |
| **pydantic** | `>=2.0` | `[core]` → promoted in Ph 2 | Phase 2 | Replaces hand-written `deserialize.py` |
| **pyyaml** | `>=6.0` | `[core]` or use stdlib `tomllib` | Phase 2 | Site config YAML files; can use `tomllib` (3.11 stdlib) + TOML to avoid extra dep |
| **pathlib** | stdlib | — | Phase 1 | Already stdlib; enforce over `os.path` |
| **enum** (`IntEnum`) | stdlib | — | Phase 1 | Replace magic `Type: int` integers |
| **tomllib** | stdlib (≥3.11) | — | Phase 2 | TOML config files; no extra dep on 3.11+ |
| **importlib.metadata** | stdlib (≥3.8) | — | Phase 1 | `__version__` without extra dep |
| **importlib.resources** | stdlib (≥3.9) | — | Phase 2 | Bundle `data/sites.yaml` inside the package |
| **httpx** | `>=0.27` | `[core]` | Phase 3b | Replaces `requests`; sync + async capable |
| **httpx[http2]** | `>=0.27` | `[api]` | Phase 3b | HTTP/2 support for the API client |
| **respx** | `>=0.21` | `[dev]` | Phase 3b | httpx mock library for API tests |
| **pyarrow** | `>=15.0` | `[parquet]` | Phase 3 | Parquet read/write via `pandas.to_parquet` |
| **panel** | `>=1.4` | `[dashboard]` | Phase 4 | Dashboard framework |
| **hvplot** | `>=0.10` | `[dashboard]` | Phase 4 | Declarative pandas `.hvplot` accessor |
| **bokeh** | `>=3.4` | `[dashboard]` | Phase 4 | Rendering backend required by panel/hvplot |
| **holoviews** | `>=1.19` | `[dashboard]` | Phase 4 | HoloViews underlies hvplot |
| **nbformat** | `>=5.10` | `[notebook]` | Phase 4 | Generate `.ipynb` files programmatically |
| **jupyter_client** | `>=8.0` | `[notebook]` | Phase 4 | Notebook kernel validation |
| **python_magnetcooling** | `>=0.2.0` | `[cooling]` | Migration prompt | Replaces `python_magnetrun/cooling/`; hydraulic fitting |
| **python_magnetapi** | — | external service | Phase 3b | REST API server (not a Python dep; must be running) |

---

## Packages to **remove**

| Package | Currently in | Remove in | Reason |
|---------|-------------|-----------|--------|
| `iapws>=1.3.4` | `[core]` | Migration prompt | Moves to `python_magnetcooling` |
| `nlopt>=2.7.0` | `[core]` | Migration prompt | Used only by cooling optimiser |
| `ht>=0.1.55` | `[system]` extras | Migration prompt | Moves to `python_magnetcooling` |
| `requests` (PyPI) | implicit (shadowed by internal module) | Phase 2 | Shadowed by `python_magnetrun/requests/` which is renamed to `fetchers/` in Phase 2; replaced by `httpx` |

---

## Dependency groups — target `pyproject.toml` shape

```toml
[project.dependencies]
# Core — required for any import of python_magnetrun
numpy        = ">=1.26"
pandas       = ">=2.1"
pydantic     = ">=2.0"        # Phase 2+
httpx        = ">=0.27"       # Phase 3b+
statsmodels  = ">=0.14"
scipy        = ">=1.12"
matplotlib   = ">=3.8"
pwlf         = ">=2.2"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
    "respx>=0.21",           # Phase 3b — mock httpx in tests
]

cooling = [
    "python-magnetcooling>=0.2.0",   # Migration prompt
]

api = [
    "httpx[http2]>=0.27",
    "pydantic>=2.0",
]

parquet = [
    "pyarrow>=15.0",                  # Phase 3
]

dashboard = [
    "panel>=1.4",                     # Phase 4
    "hvplot>=0.10",
    "bokeh>=3.4",
    "holoviews>=1.19",
]

notebook = [
    "nbformat>=5.10",                 # Phase 4
    "jupyter_client>=8.0",
]
```

---

## Phase-by-phase breakdown

### Phase 1 — Quick Wins (Weeks 1–3)

No new runtime dependencies. Changes rely entirely on Python stdlib and
packages already present:

| What | Package | stdlib? |
|------|---------|---------|
| `DataType` enum | `enum.IntEnum` | ✓ stdlib |
| `__version__` attribute | `importlib.metadata` | ✓ stdlib (3.8+) |
| `pathlib.Path` adoption | `pathlib` | ✓ stdlib |
| `logger.*` instead of `print` | `logging` | ✓ stdlib |
| CI linting gate | **ruff** `>=0.4` | add to `[dev]` |
| CI test runner | **pytest** `>=8.0` | already used; add to `[dev]` |
| Coverage | **pytest-cov** `>=5.0` | add to `[dev]` |

> **No `pyproject.toml` runtime dep changes required in Phase 1.**
> Only `[dev]` group additions.

---

### Phase 2 — Architecture (Weeks 4–8)

| What | Package | Group | Mandatory? |
|------|---------|-------|-----------|
| Pydantic models (MRecord, GObject, HMagnet) | **pydantic** `>=2.0` | `[core]` | **Yes** — replaces `deserialize.py` |
| YAML site config | **pyyaml** `>=6.0` | `[core]` | **Yes** unless TOML format chosen |
| TOML config (alternative to YAML) | `tomllib` | stdlib (3.11) | ✓ stdlib — no extra dep |
| Bundle `data/sites.yaml` in package | `importlib.resources` | stdlib (3.9) | ✓ stdlib |
| Type annotations (markers only) | `mypy` `>=1.10` | `[dev]` | Recommended |

> **Choice:** Use `tomllib` + TOML format for site config to avoid adding `pyyaml`
> as a core dependency. TOML is available in stdlib from Python 3.11. If 3.10
> support is needed, add `tomli>=2.0` as a thin fallback dep.

---

### Phase 3 — Extensibility (Weeks 9–14)

| What | Package | Group | Mandatory? |
|------|---------|-------|-----------|
| Parquet export (`saveData`) | **pyarrow** `>=15.0` | `[parquet]` | Optional extra — not in core |
| HDF5 export (`saveData`) | `tables` or `h5py` | `[hdf5]` | Optional extra |
| Plugin registries | stdlib only | — | ✓ No new deps |
| `DataLoader` protocol | stdlib `typing.Protocol` | — | ✓ No new deps |

> `pyarrow` is optional. `saveData(..., fmt="parquet")` raises a clear
> `ImportError` when `pyarrow` is not installed.

---

### Phase 3b — API Integration (Weeks 9–14, parallel with Phase 3)

| What | Package | Group | Mandatory? |
|------|---------|-------|-----------|
| HTTP client for `MagnetAPIClient` | **httpx** `>=0.27` | `[core]` | **Yes** — added to core; sync client used everywhere |
| HTTP/2 support | **httpx[http2]** | `[api]` | Optional extra |
| Pydantic response models | **pydantic** `>=2.0` | `[core]` | **Yes** (already from Phase 2) |
| httpx test mocking | **respx** `>=0.21` | `[dev]` | **Yes** — needed for all API tests |
| Credential config | `tomllib` | stdlib | ✓ No new dep |
| `~/.config/magnetrun/api.toml` parsing | `tomllib` | stdlib | ✓ No new dep |

> **Why `httpx` over `requests`?**
> - The internal `python_magnetrun/requests/` module shadows the `requests` PyPI package,
>   causing import confusion. Renaming the module (Phase 2) makes room.
> - `httpx` has a near-identical API to `requests` but supports both sync and async,
>   HTTP/2, and does not conflict with the renamed internal module.

---

### Phase 4 — Dashboards (Weeks 15–20)

All dashboard dependencies are **optional extras**. The package imports cleanly
without them; dashboard functions raise a clear `ImportError` pointing to
`pip install python-magnetrun[dashboard]`.

| What | Package | Group | Mandatory? |
|------|---------|-------|-----------|
| Dashboard framework | **panel** `>=1.4` | `[dashboard]` | Optional — `[dashboard]` extra |
| Declarative plot accessor | **hvplot** `>=0.10` | `[dashboard]` | Optional |
| Rendering backend | **bokeh** `>=3.4` | `[dashboard]` | Optional (required by panel) |
| HoloViews core | **holoviews** `>=1.19` | `[dashboard]` | Optional (required by hvplot) |
| Notebook generation | **nbformat** `>=5.10` | `[notebook]` | Optional |
| Notebook kernel API | **jupyter_client** `>=8.0` | `[notebook]` | Optional |

> **Why Panel + hvplot over Dash or Streamlit?**
> - Panel integrates natively with HoloViews/hvplot, which are already the standard
>   in the scientific Python ecosystem alongside matplotlib.
> - Dashboards defined with Panel can be used both in Jupyter notebooks (`.servable()`)
>   and as standalone web apps (`panel serve`).
> - Bokeh, Panel, hvplot, and HoloViews share the same maintainer (Anaconda/HoloViz),
>   ensuring version compatibility.

---

### Migration prompt — `python_magnetcooling` separation

| What | Package | Group | Mandatory? |
|------|---------|-------|-----------|
| Hydraulic/thermal computations | **python-magnetcooling** `>=0.2.0` | `[cooling]` | **Yes** — once `python_magnetrun/cooling/` is deleted |

---

## Minimum Python version

All phases require **Python ≥ 3.11** to use:
- `tomllib` (stdlib, replaces `tomli` package)
- `match` statements (`match fmt:` in `saveData`)
- `str | None` union syntax in type annotations
- `importlib.resources.files()` stable API

---

## Standard-library modules used (no extra install needed)

| Module | Used for | Available since |
|--------|----------|----------------|
| `enum` | `DataType(IntEnum)` | 3.4 |
| `logging` | Replace `print()` calls | always |
| `pathlib` | Replace `os.path.*` | 3.4 |
| `dataclasses` | `HydraulicData`, `SiteConfig` | 3.7 |
| `importlib.metadata` | `__version__` | 3.8 |
| `importlib.resources` | Bundled YAML data files | 3.9 (stable API 3.9) |
| `tomllib` | TOML config parsing | 3.11 |
| `typing.Protocol` | `DataLoader`, `StatPlugin` | 3.8 |
| `typing` (`runtime_checkable`) | Protocol isinstance checks | 3.8 |
| `functools` | `@lru_cache` on config loaders | always |
| `tempfile` | Temp files in API client | always |
