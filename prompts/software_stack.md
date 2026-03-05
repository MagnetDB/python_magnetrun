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
| **plotly** | `>=5.22` | `[dashboard]` | Phase 4 | Core figure library |
| **dash** | `>=2.17` | `[dashboard]` | Phase 4 | Web framework serving `FigureResampler` callbacks |
| **plotly-resampler** | `>=0.10` | `[dashboard]` | Phase 4 | Dynamic MinMaxLTTB resampling per viewport — handles 10 M+ point FEPC data |
| **tsdownsample** | `>=0.1.3` | `[dashboard]` | Phase 4 | LTTB backend; pulled in by plotly-resampler |
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
| `requests` (PyPI) | implicit (shadowed by internal module) | Phase 2 → Phase 3b | **Order matters:** rename the internal `python_magnetrun/requests/` module to `fetchers/` first (Phase 2, Task 2.1), then remove the PyPI `requests` dep and add `httpx` (Phase 3b). Doing it in reverse order leaves the shadow in place. |

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
    "plotly>=5.22",                   # Phase 4
    "dash>=2.17",
    "plotly-resampler>=0.10",         # dynamic resampling for large time series
    "tsdownsample>=0.1.3",            # MinMaxLTTB backend
]

notebook = [
    "nbformat>=5.10",                 # Phase 4
    "jupyter_client>=8.0",
    "plotly-resampler>=0.10",         # FigureWidgetResampler in notebooks
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

> **Why `httpx` over `requests`? And what order?**
>
> 1. **Phase 2 first — rename the internal module.** `python_magnetrun/requests/`
>    shadows the PyPI `requests` package. Rename it to `python_magnetrun/fetchers/`
>    (Task 2.1) and add a `DeprecationWarning` shim. Do this before touching any
>    HTTP dependency, otherwise the shadow is still in place when `httpx` is added.
>
> 2. **Phase 3b — swap to `httpx`.** Once the internal shadow is gone, remove the
>    PyPI `requests` dependency (if it was ever explicit) and add `httpx>=0.27`.
>    `httpx` has a near-identical API to `requests` but also supports async and HTTP/2.
>
> Doing it in the wrong order (removing `requests` before renaming the internal
> module) would leave user code that does `from python_magnetrun.requests import …`
> broken with no migration path.

---

### Phase 4 — Dashboards (Weeks 15–20)

All dashboard dependencies are **optional extras**. The package imports cleanly
without them; dashboard functions raise a clear `ImportError` pointing to
`pip install python-magnetrun[dashboard]`.

#### Chosen stack: Plotly + Dash + plotly-resampler

The presence of **large time series** (FEPC hybrid data at 10 kHz → 10 M+ points per
run) rules out purely static downsampling approaches. `plotly-resampler` provides
**dynamic server-side aggregation via MinMaxLTTB**: only the ~1 000 points visible in
the current viewport are sent to the browser, and the dataset is re-aggregated on every
pan or zoom — without any pre-processing step. This makes it the right tool for the
hybrid-monitor dashboard in particular.

| What | Package | Version | Group | Mandatory? |
|------|---------|---------|-------|-----------|
| Core plot library | **plotly** | `>=5.22` | `[dashboard]` | **Yes** — base for all figures |
| Dashboard web framework | **dash** | `>=2.17` | `[dashboard]` | **Yes** — serves `FigureResampler` callbacks |
| Dynamic time-series resampling | **plotly-resampler** | `>=0.10` | `[dashboard]` | **Yes** — essential for large time series |
| LTTB/MinMaxLTTB backend | **tsdownsample** | `>=0.1.3` | `[dashboard]` | Pulled in by plotly-resampler |
| Notebook widget resampling | `FigureWidgetResampler` | (part of plotly-resampler) | `[dashboard]` | ✓ No extra dep |
| Notebook generation | **nbformat** | `>=5.10` | `[notebook]` | Optional |
| Notebook kernel API | **jupyter_client** | `>=8.0` | `[notebook]` | Optional |

#### `FigureResampler` vs `FigureWidgetResampler`

| Mode | Class | When to use |
|------|-------|-------------|
| Standalone web app / `magnetrun-dashboard` CLI | `FigureResampler` | Runs a Dash server; resampling via server-side Dash callbacks; supports deployment |
| Jupyter notebook | `FigureWidgetResampler` | Uses IPython widget events and the running kernel; no port forwarding needed |

Both wrap a standard `plotly.graph_objects.Figure` and add dynamic aggregation
transparently. The same figure-building code works with both classes.

#### Comparison with Panel + hvplot

| Criterion | Plotly + Dash + plotly-resampler | Panel + hvplot + Bokeh |
|-----------|----------------------------------|----------------------|
| **Large time series** (>1 M pts) | **Native** — MinMaxLTTB per viewport, server-side | Manual downsampling required before render |
| **Hybrid FEPC data** (10 kHz, 10 M+ pts) | **Handles natively** | Needs pre-aggregation step, no live zoom resampling |
| Pupitre data (1 Hz, <100 K pts) | Fine | Fine |
| Jupyter notebook interactivity | `FigureWidgetResampler` (widget) | `.servable()` (Panel server) |
| Standalone web app | `magnetrun-dashboard` with Dash | `panel serve` |
| Ecosystem maturity | Plotly/Dash: industry standard | HoloViz: scientific Python niche |
| Production deployment | Strong (Plotly Dash is production-grade) | Moderate |
| Learning curve | Moderate — callbacks for interactivity | Low — declarative `.hvplot` accessor |

**Decision:** Use **Plotly + Dash + plotly-resampler** as the primary stack. The ability
to view FEPC hybrid data at full resolution with live pan/zoom resampling is a hard
requirement that Panel/hvplot cannot meet without bespoke pre-aggregation.

#### `pyproject.toml` extras — updated for Phase 4

```toml
[project.optional-dependencies]
dashboard = [
    "plotly>=5.22",
    "dash>=2.17",
    "plotly-resampler>=0.10",
    "tsdownsample>=0.1.3",      # pulled in by plotly-resampler; pin for reproducibility
]
notebook = [
    "nbformat>=5.10",
    "jupyter_client>=8.0",
    "plotly-resampler>=0.10",   # FigureWidgetResampler for notebook use
]
```

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
