# python_magnetrun — Improvement Plan

Synthesis of two code reviews (March 2026).
This document is the single reference for all planned improvements to the package.

---

## Context

`python_magnetrun` is a scientific data analysis package for high-field magnet facility runs
(M8, M9, M10 sites). It ingests data from three acquisition systems:

- **Pupitre** — control system text/CSV files
- **PigBrother** — TDMS monitoring files (Overview 1 Hz, Archive 1/120 Hz)
- **FEPC Hybrid** — high-frequency binary acquisition data (kHz, RMS, Trigger)

The package performs signal processing, statistical analysis, and visualization.
It is at **v0.2.0 (pre-alpha)** and needs to evolve toward:

- Cleaner, more maintainable structure
- Pluggable support for new data formats and analysis algorithms
- Integration with `python_magnetdb` (database) via `python_magnetapi` (REST API)
- Interactive dashboards for data exploration

---

## Guiding Principles

1. **Backward compatibility at every phase** — existing CLI entry points and `import` paths must
   not break until a deprecation cycle has completed.
2. **Single source of truth** — no configuration or mapping duplicated across modules.
3. **No silent failures** — remove bare `except` blocks; log and propagate errors clearly.
4. **Smallest change that fixes the problem** — avoid speculative abstraction; each item
   below targets a concrete, observable problem.

---

## Phase 1 — Quick Wins and Structural Fixes (Weeks 1–3)

These changes are low-risk, mostly non-breaking, and improve daily development quality.

### 1.1 Replace integer `Type` code with an `Enum`

**Problem:** `MagnetData.Type: int = 0` with the encoding in a comment (`0=Pandas, 1=Tdms,
2=Ensight`) is fragile and invisible to type checkers.

**File:** `python_magnetrun/magnetdata.py:40`

```python
# Add near the top of magnetdata.py
from enum import IntEnum

class DataType(IntEnum):
    PANDAS  = 0
    TDMS    = 1
    ENSIGHT = 2
```

Replace all `if self.Type == 0:` checks with `if self.Type == DataType.PANDAS:`.
Keep integer values so that existing serialised data is unaffected.

---

### 1.2 Replace `print()` with `logger.*` in core modules

**Problem:** `MagnetRun.py` (lines 26–28, 110–113, 119–130) and `magnetdata.py` (line 57)
use `print()` instead of the already-initialised `logger`.

**Fix:** `print(...)` → `logger.debug(...)` or `logger.info(...)` throughout all core modules.

---

### 1.3 Remove module-level matplotlib side effects

**Problem:** `matplotlib.rcParams["text.usetex"] = True` at module level in
`python_magnetrun.py`, `outliers.py`, and `pupitre.py` breaks import in any non-LaTeX
environment (e.g., CI servers, Windows, the API backend).

**Fix:** Move LaTeX configuration inside plot functions, guarded by a keyword argument:

```python
def plotData(self, ..., use_latex: bool = False):
    if use_latex:
        matplotlib.rcParams["text.usetex"] = True
```

---

### 1.4 Always set `self.data` in `MagnetData.__init__`

**Problem:** `if Data is not None: self.Data = Data` leaves `self.Data` undefined when
`Data=None`, producing `AttributeError` instead of a clear message downstream.

**Fix:**

```python
self.data: pd.DataFrame | dict | None = Data
```

Add a private helper used by all methods that require data:

```python
def _require_data(self) -> pd.DataFrame | dict:
    if self.data is None:
        raise RuntimeError("MagnetData has no loaded data")
    return self.data
```

---

### 1.5 Rename `test-*.py` files to `test_*.py`

**Problem:** Hyphenated filenames cannot be imported as Python modules, which breaks
pytest plugins, `coverage`, and `python -m` invocations.

**Fix:** Rename all `test-*.py` → `test_*.py` under `tests/`.
Remove the `"test-*.py"` pattern from `[tool.pytest.ini_options]` in `pyproject.toml`.

---

### 1.6 Remove commented-out dead code

Examples identified:

| File | Lines | Description |
|------|-------|-------------|
| `MRecord.py` | 116–127 | Commented-out `__le__` and `__ge__` methods |
| `MagnetRun.py` | 63 | `# data.removeData(...)` |
| `magnetdata.py` | 71–74 | Multiple `# print(...)` debug blocks |

**Rule:** Keep a comment only when it explains *why*, not *what*. Track removal via a GitHub
issue if unsure.

---

### 1.7 Fix absolute developer paths in `analysis/config.py`

**Problem:** Lines 117–134 default to `/home/LNCMI-G/christophe.trophime/...` — fails for
every other user who has not set env vars.

**Fix:** Default to a sensible user-space location:

```python
DEFAULT_DATA_DIR: str = os.environ.get(
    "MAGNETRUN_DATA_DIR",
    str(Path.home() / ".local" / "share" / "magnetrun" / "data"),
)
```

Document env vars `MAGNETRUN_DATA_DIR` and `MAGNETRUN_PIGBROTHER_DATA_DIR` in `README.md`.

---

### 1.8 Fix `getInsert()` path logic in `MagnetRun.py`

**Problem:** The current logic strips extensions incorrectly for paths with directories.

**Fix:**

```python
def getInsert(self) -> str:
    return Path(self.MagnetData.FileName).stem
```

---

### 1.9 Add `__all__` to `python_magnetrun/__init__.py`

Explicitly declare the public API. Controls `from python_magnetrun import *` and
improves IDE discoverability.

---

### 1.10 Add CI workflow

Create `.github/workflows/test.yml`:

- Trigger: `push`, `pull_request` on `master` and `claude/*` branches
- Steps: install dev deps → `ruff check .` → `pytest tests/`
- Optional: upload coverage to Codecov

---

## Phase 2 — Architecture and Maintainability (Weeks 4–8)

### 2.1 Rename `requests/` module → `fetchers/`

**Problem:** `python_magnetrun/requests/` shadows the popular `requests` PyPI package, causing
import confusion for tools and IDEs.

**Fix:**

```
python_magnetrun/requests/  →  python_magnetrun/fetchers/
```

Update all internal imports (`MRecord.py`, `python_magnetrun.py`, etc.).
Keep a one-release shim `python_magnetrun/requests/__init__.py` with a deprecation warning
pointing to `fetchers/`.

---

### 2.2 Rename `python_magnetrun/python_magnetrun.py` → `cli_main.py`

**Problem:** File name matches package name — ambiguous to navigate and confusing for
static analysis tools.

**Fix:** Rename to `python_magnetrun/cli_main.py`.
Update `pyproject.toml`:

```toml
[project.scripts]
python-magnetrun = "python_magnetrun.cli_main:main"
```

---

### 2.3 Split `magnetdata.py` (1 300+ lines) into focused submodules

Proposed structure:

```
python_magnetrun/magnetdata/
├── __init__.py        # re-exports MagnetData — zero breaking changes
├── core.py            # MagnetData class definition + __init__, __repr__
├── loaders.py         # fromtxt(), fromcsv(), fromtdms(), fromStringIO()
├── transforms.py      # addData(), renameData(), removeData(), cleanupData()
└── stats.py           # stats(), getStats(), summary methods
```

Keep `from python_magnetrun.magnetdata import MagnetData` working throughout.

---

### 2.4 Split `python_magnetrun.py` CLI (44 KB) into subcommand modules

Proposed structure:

```
python_magnetrun/commands/
├── __init__.py
├── info.py    # --info, --list
├── plot.py    # --plot, --vs-time, --key-vs-key
├── stats.py   # --stats, --plateau
├── export.py  # --save, --output
└── main.py    # top-level argparse router
```

---

### 2.5 Centralize housing/site configuration

**Problem:** Housing-specific logic (M8/M9/M10 channel renames, current aggregation) is
duplicated between `MagnetRun.prepareData()` and `analysis/config.py:SITE_CONFIGS`.

**Fix:**

1. Extend `SiteConfig` in `analysis/config.py` with channel rename rules and current
   aggregation fields.
2. `prepareData()` receives a `SiteConfig` object and derives all renames from it.
3. Delete duplicated mappings from `prepareData_legacy()`.

`SITE_CONFIGS` becomes the **single source of truth** for site topology.

---

### 2.6 Support YAML/TOML site configuration files

**Problem:** Adding a new site (M1, M5, M7, M11) requires editing library source code.

**Fix:** Load site configs from:

1. `~/.config/magnetrun/sites.yaml` (user override)
2. Package-bundled `python_magnetrun/data/sites.yaml` (defaults)

```yaml
# sites.yaml
M9:
  reference_gr1_current: IH
  reference_gr2_current: IB
  flow_mapping: {Flow1: FlowH, Flow2: FlowB}
  voltage_channels_gr1: [UH]
  voltage_channels_gr2: [UB, Ucoil15, Ucoil16]
M7:
  reference_gr1_current: IH
  ...
```

`get_site_config()` checks user config first, falls back to built-ins.
Use `tomllib` (stdlib, Python ≥ 3.11) or `pyyaml` depending on chosen format.

---

### 2.7 Replace bare `except` blocks

**Problem:** `MagnetRun.fromStringIO()` catches all exceptions silently, writes
`wrongdata.txt`, and continues — hides bugs.

**Fix:** Catch only specific exceptions (`ValueError`, `pd.errors.ParserError`),
log with `logger.exception()`, and re-raise or return `None` with documentation.

---

### 2.8 Adopt `pathlib.Path` consistently

**Problem:** Mixed use of `os.path`, `os.path.splitext`, and string concatenation for paths
across `magnetdata.py`, `MagnetRun.py`, `MRecord.py`, `fetchers/cli.py`.

**Fix:** Systematically replace `os.path.*` calls with `pathlib.Path` equivalents.

---

### 2.9 Replace hand-written serialization in `deserialize.py`

**Problem:** Custom `serialize_instance()` is brittle and hard to maintain.

**Fix (preferred):** Migrate `MRecord`, `GObject`, `HMagnet` to `pydantic.BaseModel`:

- Provides validation, JSON schema generation, and direct API compatibility.
- `model.model_dump_json()` replaces custom serialization.
- Pydantic v2 is a natural fit given the REST API integration plan.

**Fix (minimal):** Use `@dataclass` + `dataclasses.asdict()` if pydantic is not adopted.

---

### 2.10 Split `processing/hysteresis.py` (37 KB)

Proposed split:

```
python_magnetrun/processing/hysteresis/
├── __init__.py        # re-exports for backwards compat
├── analysis.py        # Loop detection algorithms
├── fitting.py         # Linear, quadratic, exponential, power-law models
├── plotting.py        # Visualization functions
└── outliers.py        # Hysteresis-specific outlier removal
```

---

### 2.11 Add type annotations and configure mypy

- Add `py.typed` marker file (PEP 561).
- Add to `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
strict = false
```

- Annotate all public APIs in `magnetdata.py`, `MagnetRun.py`, `MRecord.py` as the files
  are touched during other refactors.

---

### 2.12 Configure ruff

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]   # pycodestyle, pyflakes, isort, pyupgrade, bugbear
ignore  = ["E501"]                      # line length handled by formatter
```

---

## Phase 3 — Extensibility (Weeks 9–14)

### 3.1 Introduce a `DataLoader` protocol

**File to create:** `python_magnetrun/protocols.py`

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DataLoader(Protocol):
    @classmethod
    def can_load(cls, path: str) -> bool: ...
    @classmethod
    def load(cls, path: str, **kwargs) -> "MagnetData": ...
    def get_format_name(self) -> str: ...
```

Generalises the existing pattern in `hybrid/data_protocol.py`.

---

### 3.2 Format auto-detection factory method

Add to `MagnetData`:

```python
_LOADER_REGISTRY: dict[str, type[DataLoader]] = {
    ".txt":  PupitreLoader,
    ".csv":  CsvLoader,
    ".tdms": TdmsLoader,
}

@classmethod
def from_file(cls, path: str, **kwargs) -> "MagnetData":
    suffix = Path(path).suffix.lower()
    loader = _LOADER_REGISTRY.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported format: {suffix!r}. "
                         f"Supported: {list(_LOADER_REGISTRY)}")
    return loader.load(path, **kwargs)
```

Adding a new format = writing one new `DataLoader` class and one registry entry.
No changes to `MagnetData` core.

---

### 3.3 Statistics plugin registry

**Problem:** Adding new statistics requires modifying `getStats()` directly.

**File:** `python_magnetrun/processing/registry.py`

```python
_STAT_PLUGINS: dict[str, Callable[[pd.DataFrame], Any]] = {}

def register_stat(name: str):
    def decorator(fn: Callable) -> Callable:
        _STAT_PLUGINS[name] = fn
        return fn
    return decorator

# Usage:
@register_stat("entropy")
def compute_entropy(df: pd.DataFrame) -> float:
    return float(scipy.stats.entropy(df.values.flatten()))
```

`getStats(field, extra=["entropy"])` runs all registered plugins and merges results.

---

### 3.4 Analysis algorithm registry

Same pattern for smoothers, detectors, and outlier methods:

```python
_SMOOTHERS:  dict[str, Callable] = {}
_DETECTORS:  dict[str, Callable] = {}

def register_smoother(name: str): ...
def register_detector(name: str): ...
```

CLI `--smoother savgol` dispatches via `_SMOOTHERS["savgol"]`.
Users can register their own algorithms without modifying library source.

---

### 3.5 `addData` callable support

**Problem:** Formula string approach (`addData("IH_ref", "IH_ref = Idcct1 + Idcct2")`) is
fragile, hard to test, and invisible to type checkers.

**Fix:** Accept both a string formula (legacy) and a callable:

```python
def addData(
    self,
    key: str,
    formula: str | Callable[[pd.DataFrame], pd.Series],
) -> None:
    if callable(formula):
        self.data[key] = formula(self.data)
    else:
        # existing string eval path
        ...
```

---

### 3.6 Add Parquet and HDF5 export to `saveData`

```python
def saveData(self, filename: str, format: str = "csv") -> None:
    match format:
        case "csv":     self.data.to_csv(filename)
        case "parquet": self.data.to_parquet(filename)
        case "hdf5":    self.data.to_hdf(filename, key="magnetrun")
        case _:         raise ValueError(f"Unknown format: {format!r}")
```

Parquet is particularly useful for large FEPC runs and for data shared with `python_magnetdb`.

---

### 3.7 Expand test coverage

| File to add | Content |
|-------------|---------|
| `tests/test_magnetdata.py` | `fromtxt()`, `fromcsv()`, `addData()`, `renameData()`, `getData()` |
| `tests/test_magnetrun.py` | `fromtxt()`, `fromcsv()`, `prepareData()`, `getStats()` |
| `tests/test_mrecord.py` | `to_json()`, `__eq__()`, `getDataFilename()` |
| `tests/test_processing.py` | Smoothers, breakpoints, plateau detection |

Use fixtures with sample data files from `data/`.

---

## Phase 4 — `python_magnetapi` Integration (Weeks 9–14, parallel)

### 4.1 Create `python_magnetrun/api/` client module

```
python_magnetrun/api/
├── __init__.py
├── client.py      # MagnetAPIClient
├── models.py      # Pydantic response models
├── auth.py        # Token/credential management
└── cli.py         # magnetrun-api subcommands
```

#### `client.py` outline

```python
import httpx
from python_magnetrun.api.models import MagnetRunRecord, RunStats

class MagnetAPIClient:
    def __init__(
        self,
        base_url: str  = os.environ.get("MAGNETAPI_URL", "http://localhost:8000"),
        api_key: str | None = os.environ.get("MAGNETAPI_KEY"),
    ): ...

    def get_run(self, run_id: int) -> MagnetRunRecord: ...
    def list_runs(self, site: str | None = None, ...) -> list[MagnetRunRecord]: ...
    def upload_run(self, run: MagnetRun) -> int: ...          # returns assigned run_id
    def post_stats(self, run_id: int, stats: dict) -> None: ...
    def post_anomalies(self, run_id: int, anomalies: list) -> None: ...
```

Credentials via environment variables: `MAGNETAPI_URL`, `MAGNETAPI_KEY`.
Use `httpx` instead of `requests` for async-capable HTTP.

---

### 4.2 Pydantic API models in `api/models.py`

```python
from pydantic import BaseModel
from datetime import datetime

class MagnetRunRecord(BaseModel):
    id: int
    site: str
    housing: str
    start_time: datetime
    end_time: datetime
    file_url: str

class RunStats(BaseModel):
    run_id: int
    field_max: float
    ih_mean: float
    ib_mean: float
    plateau_count: int
    anomaly_count: int
```

---

### 4.3 Local cache for downloaded files

```python
class MagnetAPIClient:
    def __init__(self, ..., cache_dir: Path = Path.home() / ".cache" / "magnetrun"):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_run_data(self, run_id: int) -> MagnetRun:
        cached = self.cache_dir / f"{run_id}.parquet"
        if cached.exists():
            return MagnetRun.from_parquet(cached)
        # download, cache, return
```

---

### 4.4 Migrate `MRecord.getData()` to use API when available

**Current:** `MRecord.getData()` directly scrapes `fetchers/connect.py` (srv-data endpoint).

**Proposed:** Check for `MAGNETAPI_URL`; if set, use `MagnetAPIClient.get_run_data()`.
Fall back to direct scraping for backwards compatibility.

---

### 4.5 Upload results back to magnetdb

Add to `MagnetRun`:

```python
def upload(self, client: MagnetAPIClient) -> int:
    """Serialize run + signature + stats, POST to /api/runs/. Returns run_id."""
```

This enables a closed loop:

```
File on disk → MagnetRun → analysis → upload stats → magnetdb
                                    ↑
                              API retrieval
```

---

## Phase 5 — Interactive Dashboards (Weeks 15–20)

### 5.1 Restructure `panels/` → `dashboards/`

```
python_magnetrun/dashboards/
├── __init__.py
├── run_overview.py      # Time-series overview (field, currents, flow, temperatures)
├── field_analysis.py    # Field vs current, B-I curves
├── comparison.py        # Multi-run overlay (time-aligned)
├── hybrid_monitor.py    # Real-time / replay FEPC data view
└── widgets.py           # Shared Panel widgets (time range slider, key selector)
```

All dashboards importable as Python objects — not just runnable scripts.

---

### 5.2 `run_overview` dashboard

**Technology:** `panel` + `hvplot`

```python
import panel as pn
import hvplot.pandas

def run_overview_dashboard(run: MagnetRun) -> pn.viewable.Viewable:
    df = run.getData()
    current_plot = df.hvplot.line(x="t", y=["IH", "IB"], title="Currents (A)")
    field_plot   = df.hvplot.line(x="t", y="Field",      title="Field (T)")
    flow_plot    = df.hvplot.line(x="t", y=["FlowH", "FlowB"], title="Flow (l/s)")

    # Overlay regime markers from Signature
    regime_panel = _build_regime_overlay(run)

    return pn.Column(field_plot, current_plot, flow_plot, regime_panel)
```

Controls: time-range slider, field-key checkboxes, smoothing toggle.

---

### 5.3 `comparison` dashboard

- Load multiple `MagnetRun` objects from files or via `MagnetAPIClient`.
- Normalize time axes to run start.
- Overlay field profiles.
- Statistics table per run (plateau count, max field, anomaly count).

---

### 5.4 CLI entry point for dashboards

Add to `pyproject.toml`:

```toml
[project.scripts]
magnetrun-dashboard = "python_magnetrun.dashboards.cli:main"
```

```
magnetrun-dashboard overview data/run_20240315.txt
magnetrun-dashboard compare data/run_A.txt data/run_B.txt
magnetrun-dashboard hybrid  data/hybrid_dir/
```

Serves via `panel serve`; opens browser automatically.

---

### 5.5 Jupyter notebook auto-generation

Add a `magnetrun-to-notebook` CLI that generates a pre-filled `.ipynb` from a data file:

```
magnetrun-to-notebook data/run_20240315.txt --output analysis.ipynb
```

Uses `nbformat` to programmatically create notebooks with standard analysis cells:
data loading, key listing, time-series plot, stats summary, plateau detection.

---

## Summary: Critical Files and Changes

| Priority | File / Path | Change |
|----------|-------------|--------|
| **High** | `magnetdata.py` | `DataType` enum, always-set `self.data`, split into submodule |
| **High** | `MagnetRun.py` | `print()` → `logger`, centralize housing config |
| **High** | `python_magnetrun.py` | Rename to `cli_main.py`, split into `commands/` |
| **High** | `requests/` | Rename to `fetchers/` |
| **High** | `tests/test-*.py` | Rename to `test_*.py`, add CI workflow |
| **High** | `analysis/config.py` | Fix absolute paths; make `SITE_CONFIGS` single truth |
| **Medium** | `processing/hysteresis.py` | Split into 3–4 sub-files |
| **Medium** | `deserialize.py` | Replace with `pydantic.BaseModel` |
| **Medium** | `panels/` | Restructure into importable `dashboards/` module |
| **Medium** | All modules | `pathlib.Path` consistently; remove bare `except` |
| **New** | `protocols.py` | `DataLoader` protocol |
| **New** | `api/` | `MagnetAPIClient`, Pydantic models, auth, CLI |
| **New** | `dashboards/` | Panel/hvplot dashboards |
| **New** | `data/sites.yaml` | Externalised site configuration |
| **New** | `processing/registry.py` | Stat and algorithm plugin registries |

---

## Recommended Libraries to Adopt

| Purpose | Library | Already used? |
|---------|---------|---------------|
| Data validation / serialization | `pydantic v2` | No |
| Linting + formatting | `ruff` | Configured, enforce in CI |
| Type checking | `mypy` | No |
| Dashboard framework | `panel` + `hvplot` | Partial (`panels/`) |
| Config files | `tomllib` (stdlib 3.11+) or `pyyaml` | No |
| REST API client | `httpx` (async-capable) | No (replace `requests`) |
| Notebook generation | `nbformat` | No |
| Property-based testing | `hypothesis` | No |

---

## Migration Phases — Summary

| Phase | Scope | Weeks |
|-------|-------|-------|
| **1** | Enum, print→logger, matplotlib side-effects, path fixes, CI, test renames | 1–3 |
| **2** | Rename `requests/`→`fetchers/`, rename `python_magnetrun.py`, split `magnetdata.py`, centralize config, `pathlib`, ruff, mypy | 4–8 |
| **3** | `DataLoader` protocol, `from_file()` auto-detect, stat/algo registries, Parquet export, expanded tests, `addData` callables | 9–14 |
| **3b** | `api/` client, Pydantic models, cache, `MRecord` API fallback, `upload()` | 9–14 |
| **4** | `dashboards/` module, `run_overview`, `comparison`, `magnetrun-dashboard` CLI, notebook generator | 15–20 |

Each phase ends with **all existing tests green** and **existing CLI entry points unchanged**.

---

## Verification Plan

1. **Unit tests:** `pytest tests/` — all existing and new tests pass.
2. **Import smoke test:**
   ```
   python -c "from python_magnetrun import MagnetData, MagnetRun, MRecord"
   ```
3. **CLI smoke tests:**
   ```
   python-magnetrun --help
   python-magnetrun <datafile.txt> info --list
   python-magnetrun <datafile.txt> stats
   ```
4. **Backwards compat:** All scripts in `examples/` run without modification.
5. **Dashboard:** `magnetrun-dashboard overview <datafile.txt>` opens browser panel.
6. **API client:** `from python_magnetrun.api import MagnetAPIClient` imports cleanly
   and can be instantiated with a mock URL.
7. **Linting:** `ruff check .` reports zero errors.
8. **Type checking:** `mypy python_magnetrun/` reports zero errors on annotated files.
