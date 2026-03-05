# python_magnetrun — Architecture Review & Roadmap

`python_magnetrun` is a scientific data analysis package for high-field magnet facility runs (M8, M9, M10). It ingests data from three acquisition systems (Pupitre TXT/CSV, PigBrother TDMS, FEPC hybrid kHz/RMS/trigger), performs signal processing and statistical analysis, and produces visualizations. The package is pre-alpha (v0.1.0).

---

## Implementation Progress

| # | Item | Status |
|---|---|---|
| 1.1 | Rename `requests/` → `fetchers/` | ⬜ Not started |
| 1.2 | Rename `python_magnetrun.py` → `cli_main.py` | ⬜ Not started |
| 1.3 | Replace integer type codes with `Enum` in `MagnetData` | ⬜ Not started |
| 1.4 | Replace `print()` with structured logging in core modules | ⬜ Not started |
| 1.5 | Remove commented-out dead code | ⬜ Not started |
| 1.6 | Add type annotations to core public APIs | ⬜ Not started |
| 1.7 | Split `magnetdata.py` into focused modules | ⬜ Not started |
| 1.8 | Split `python_magnetrun.py` CLI into subcommand modules | ⬜ Not started |
| 1.9 | Fix `fromtdms` file-opening anti-pattern | ⬜ Not started |
| 2.1 | Add type annotations + mypy configuration | ⬜ Not started |
| 2.2 | Expand test coverage for core modules | 🟡 Partial — `test_waterflow_pipeline.py` added; core modules still untested |
| 2.3 | Replace hand-written JSON serialization with dataclasses/pydantic | ⬜ Not started |
| 2.4 | Centralize housing/site configuration | 🟡 Partial — `prepareData()` now config-dict-driven; YAML/TOML externalization pending |
| 2.5 | Adopt `pathlib.Path` consistently | ⬜ Not started |
| 2.6 | Replace bare `except Exception:` handlers | ⬜ Not started |
| 2.7 | Use `ruff` for linting and formatting | ⬜ Not started |
| 2.8 | Reduce `hysteresis.py` complexity | ⬜ Not started |
| 3.1 | Introduce `DataLoader` Protocol | ⬜ Not started |
| 3.2 | Format auto-detection factory method | ⬜ Not started |
| 3.3 | Plugin-style analysis algorithm registry | ⬜ Not started |
| 3.4 | Statistics plugin interface | ⬜ Not started |
| 3.5 | Site configuration via YAML/TOML files | ⬜ Not started |
| 4.1 | Create `python_magnetrun/api/` client module | ⬜ Not started |
| 4.2 | `MRecord.getData()` via API | ⬜ Not started |
| 4.3 | Upload processed results to magnetdb | ⬜ Not started |
| 5.1 | Expand `panels/` into proper dashboard module | ⬜ Not started |
| 5.2 | `run_overview` dashboard | ⬜ Not started |
| 5.3 | `comparison` dashboard | ⬜ Not started |
| 5.4 | Jupyter notebook support | ⬜ Not started |
| 6.1 | Fix `getInsert()` in `MagnetRun.py` | ⬜ Not started |
| 6.2 | Add `__all__` to package `__init__.py` | ⬜ Not started |
| 6.3 | Consistent `__repr__` via dataclasses or f-strings | ⬜ Not started |
| 6.4 | Remove module-level matplotlib side effects | ⬜ Not started |
| 6.5 | Remove hard-coded developer paths | ⬜ Not started |
| 6.6 | `bilan.py` and `flow_params.py` — document purpose | ⬜ Not started |
| — | **Cooling system separation** (`cooling/` → `python_magnetcooling` submodule) | ✅ Done (`separate-cooling`) |
| — | **Waterflow pipeline** (`waterflow_pipeline.py`) | ✅ Done (`separate-cooling`) |
| — | **Flexible data preparation** config-dict API | ✅ Done (`separate-cooling`) |

---

## separate-cooling Branch Summary

> **Branch:** `separate-cooling` · **Merged into master:** 2026-03-05 · **Latest commit:** `4233d14`

This branch landed three independent improvements not originally enumerated in the review, plus a partial implementation of item 2.4.

### A. Cooling System Extracted to Submodule

The local `python_magnetrun/cooling/` package has been replaced by the external
[`python_magnetcooling`](https://github.com/MagnetDB/python_magnetcooling) git submodule.

**Deleted files:**
- `python_magnetrun/cooling/clawtest1.py`
- `python_magnetrun/cooling/heatexchanger_primary.py`
- `python_magnetrun/cooling/heatexchanger_primary_orig.py`
- `python_magnetrun/cooling/water.py`
- `python_magnetrun/cooling/wproperties.py`

**Added:** `.gitmodules` pointing to `git@github.com:MagnetDB/python_magnetcooling.git`

**Updated imports** in `python_magnetrun/bilan.py`:
```python
# Before
from .cooling import water
water.getRho(...)
water.getCp(...)

# After
from python_magnetcooling.water_properties import get_rho, get_cp
get_rho(...)
get_cp(...)
```

### B. Waterflow Pipeline (`waterflow_pipeline.py`)

New module `python_magnetrun/waterflow_pipeline.py` (~450 lines) integrates `MagnetRun`
data with `python_magnetcooling` for hydraulic analysis:

- `HydraulicData` dataclass — structured container for flow/pressure measurements
- `extract_hydraulic_data()` — extracts flow/pressure channels from a `MagnetRun`
- `detect_imax_from_plateaus()` — identifies plateau regions for pump-curve fitting

Accompanying examples:
- `examples/flow_params_magnetrun_pipeline.py` (676 lines)
- `examples/flow_params_pipeline.py` (492 lines)
- `examples/waterflow_debitbrut_example.py` (202 lines)

### C. Flexible Data Preparation (partial 2.4)

`prepareData()` and `cleanupData()` now accept explicit configuration dictionaries,
replacing the `if housing == "M9": ... elif housing in ["M8", "M10"]: ...` chains.
Legacy implementations are preserved for backwards compatibility.

**`MagnetRun.prepareData()` new signature** (`MagnetRun.py:78`):
```python
def prepareData(
    data: MagnetData,
    housing: str,
    keys_to_remove: list[str] | None = None,
    keys_to_rename: dict[str, str] | None = None,
    keys_to_add: dict[str, str] | None = None,
    debug: bool = False,
)
```

**`MagnetData.cleanupData()` new signature** (`magnetdata.py:732`):
```python
def cleanupData(
    self,
    keys_to_remove: list[str] | None = None,
    keys_to_rename: dict[str, str] | None = None,
    keys_to_add: dict[str, str] | None = None,
    debug: bool = False,
)
```

Operation order inside `cleanupData()`:
1. Apply `keys_to_add` — add computed columns (e.g. `IH_ref = Idcct1 + Idcct2`)
2. Apply `keys_to_rename` — rename columns (e.g. `Flow1 → FlowH`)
3. Apply `keys_to_remove` — drop columns
4. Standard cleanup — remove zero/empty columns, deduplicate

Smart detection: when `keys_to_rename` contains `Icoil*` mappings the legacy
auto-detection (~200 lines) is skipped entirely. When `keys_to_add` defines `UH`/`UB`
the Ucoil auto-detection is also skipped.

**Legacy methods preserved:**
- `MagnetRun.prepareData_legacy()` — original housing-specific logic
- `MagnetData.cleanupData_legacy()` — original auto-detection

**Documentation:** `CLEANUP_USAGE_EXAMPLES.md` (337 lines) with migration guide and
best practices.

### D. Test Coverage

New test file `tests/test_waterflow_pipeline.py` (428 lines) covering:
- `HydraulicData` construction and validation
- `extract_hydraulic_data()` with synthetic `MagnetRun` fixtures
- `detect_imax_from_plateaus()` edge cases

---

## 1. Readability Improvements

### 1.1 Rename `requests/` module → `fetchers/` or `datasources/`

**Problem:** `python_magnetrun/requests/` shadows the popular `requests` PyPI library,
causing potential import confusion.

**Files:** `python_magnetrun/requests/` → `python_magnetrun/fetchers/`

**Impact:** Rename imports in `MRecord.py`, `python_magnetrun.py`, any users.

> **Status:** ⬜ Not started.

---

### 1.2 Rename `python_magnetrun.py` CLI module → `cli_main.py` or `commands.py`

**Problem:** `python_magnetrun/python_magnetrun.py` matches the package name — ambiguous
and confusing to navigate.

**Files:** `python_magnetrun/python_magnetrun.py` → `python_magnetrun/cli_main.py`

**Impact:** Update `pyproject.toml` entry point: `python_magnetrun.cli_main:main`

> **Status:** ⬜ Not started.

---

### 1.3 Replace integer type codes with an `Enum` in `MagnetData`

**Problem:** `Type: int = 0` with comment `# 0=Pandas, 1=Tdms, 2=Ensight` is fragile
and undiscoverable.

**File:** `python_magnetrun/magnetdata.py` line 40

**Solution:**
```python
from enum import Enum, auto

class DataType(Enum):
    PANDAS = 0
    TDMS = 1
    ENSIGHT = 2
```

Replace all `if self.Type == 0:` checks with `if self.Type == DataType.PANDAS:`.

> **Status:** ⬜ Not started. Note: `cleanupData()` at line 757 still uses
> `if self.Type != 0:` — this check should become `if self.Type != DataType.PANDAS:`
> once the Enum is introduced.

---

### 1.4 Replace `print()` calls with structured logging in core modules

**Problem:** `MagnetRun.py` (lines 26–28, 110–113, 119–130), `magnetdata.py` (line 57)
use `print()` instead of the already-initialized `logger`.

**Fix:** Replace all `print(...)` with `logger.debug(...)` or `logger.info(...)`.

> **Status:** ⬜ Not started. The new `prepareData()` in `separate-cooling` correctly
> uses `logger.debug()` throughout — existing legacy code still has stale `print()` calls.

---

### 1.5 Remove commented-out dead code

**Examples:**
- `MRecord.py` lines 116–127: commented-out `__le__` and `__ge__` methods
- `MagnetRun.py` line 63: `# data.removeData(...)`
- `magnetdata.py` lines 71–74: several commented `# print(...)` blocks

**Action:** Remove all commented-out code blocks that have no explanatory comment.

> **Status:** ⬜ Not started. Note: `separate-cooling` introduced new commented-out
> blocks in `MagnetRun.py` (lines 126–145) where the legacy housing logic was
> commented out rather than deleted. These should be cleaned up.

---

### 1.6 Add type annotations to core public APIs

**Files lacking annotations:** `MagnetData` methods (most return `Any`), `MagnetRun`
factory methods, `prepareData()` return type.

**Priority targets:**
- `magnetdata.py`: `getData()`, `getKeys()`, `addData()`, `renameData()`
- `MagnetRun.py`: return types on all factory methods

> **Status:** ⬜ Not started. The new `prepareData()` signature uses `list[str] | None`
> and `dict[str, str] | None` — a good model for the rest of the codebase.

---

### 1.7 Split `magnetdata.py` (1300+ lines) into focused modules

**Proposed split:**

```
magnetdata/
├── core.py        # MagnetData class definition + constructors
├── loaders.py     # fromtxt(), fromcsv(), fromtdms(), fromensight(), fromStringIO()
├── transforms.py  # addData(), renameData(), removeData(), cleanupData()
├── stats.py       # stats(), getStats(), summary methods
└── __init__.py    # re-export MagnetData for backwards compat
```

> **Status:** ⬜ Not started. `separate-cooling` added ~153 lines to `magnetdata.py`
> (the new `cleanupData()` method), making the split more urgent.

---

### 1.8 Split `python_magnetrun.py` (44 KB CLI) into subcommand modules

**Proposed split under `python_magnetrun/commands/`:**

```
commands/
├── info.py    # --info, --list subcommands
├── plot.py    # --plot, --vs-time, --key-vs-key subcommands
├── stats.py   # --stats, --plateau subcommands
├── export.py  # --save, --output subcommands
└── main.py    # top-level argparse router
```

> **Status:** ⬜ Not started.

---

### 1.9 Fix `fromtdms` file-opening anti-pattern

**Problem:** `magnetdata.py` line 62 does `with open(name, "r"):` just to check the
extension — incorrect (opens as text), and extension check should be before opening.

**Fix:** Move extension check before the `with open(...)` block; or simply use
`pathlib.Path(name).suffix`.

> **Status:** ⬜ Not started.

---

## 2. Maintainability Improvements

### 2.1 Add type annotations + mypy configuration

- Add `py.typed` marker file
- Add `[tool.mypy]` section to `pyproject.toml` with `strict = false`,
  `ignore_missing_imports = true`
- Annotate all public APIs in `magnetdata.py`, `MagnetRun.py`, `MRecord.py`

> **Status:** ⬜ Not started.

---

### 2.2 Expand test coverage for core modules

Currently untested: `MagnetData`, `MagnetRun`, `MRecord`, `processing/`, `signature.py`

**Add tests in `tests/`:**
- `test_magnetdata.py`: test `fromtxt()`, `fromcsv()`, `addData()`, `renameData()`, `getData()`
- `test_magnetrun.py`: test `fromtxt()`, `fromcsv()`, `prepareData()`, `getStats()`
- `test_mrecord.py`: test `to_json()`, `__eq__()`, `getDataFilename()`
- `test_processing.py`: test smoothers, breakingpoints, plateau detection

Use fixtures with the sample data in `data/` directory.

> **Status:** 🟡 Partial. `tests/test_waterflow_pipeline.py` (428 lines) was added in
> `separate-cooling` covering the new hydraulic pipeline. Core modules (`MagnetData`,
> `MagnetRun`, `MRecord`, `processing/`) remain untested.

---

### 2.3 Replace hand-written JSON serialization with dataclasses/pydantic

**Problem:** `deserialize.py` has custom `serialize_instance()` — brittle and
unmaintainable.

**Solution A (minimal):** Convert `MRecord`, `GObject`, `HMagnet` to `@dataclass` and
use `dataclasses.asdict()`.

**Solution B (preferred):** Use `pydantic.BaseModel` — provides validation, JSON
schema, and API compatibility.

**Files:** `python_magnetrun/deserialize.py`, `MRecord.py`, `GObject.py`, `HMagnet.py`

> **Status:** ⬜ Not started.

---

### 2.4 Centralize housing/site configuration

**Problem:** Housing-specific logic (M8, M9, M10) was hard-coded in `prepareData()` in
`MagnetRun.py` (lines 40–60) and duplicated in `analysis/config.py`.

**Solution:** Extend `analysis/config.py`'s `SiteConfig` dataclass to include channel
mappings and current aggregation rules. `prepareData()` reads from config instead of
`if housing == "M9":` chains.

> **Status:** 🟡 Partial — implemented in `separate-cooling`.
>
> `prepareData()` (`MagnetRun.py:78`) and `cleanupData()` (`magnetdata.py:732`) now
> accept `keys_to_remove`, `keys_to_rename`, and `keys_to_add` dicts, eliminating
> the hardcoded housing branches from the hot path. The legacy implementation is
> preserved in `prepareData_legacy()` / `cleanupData_legacy()`.
>
> **Remaining work:**
> - The commented-out housing blocks in `prepareData()` (lines 126–145) should be
>   removed once migration is confirmed complete.
> - Housing config dicts (e.g. M9: `{IH_ref: Idcct1+Idcct2, Flow1→FlowH, ...}`)
>   should be externalised to `analysis/config.py` or a YAML file (see item 3.5)
>   rather than living at call sites.
> - `SiteConfig` dataclass extension not yet done.

---

### 2.5 Adopt `pathlib.Path` consistently

**Problem:** Mix of `os.path`, `os.path.splitext`, string concatenation for paths.

**Files:** `magnetdata.py`, `MagnetRun.py`, `MRecord.py`, `requests/cli.py`

**Fix:** Replace `os.path.*` calls with `pathlib.Path` equivalents.

> **Status:** ⬜ Not started.

---

### 2.6 Replace bare `except Exception:` handlers

**Problem:** `MagnetRun.fromStringIO()` line 169 catches all exceptions silently,
writes `wrongdata.txt`, and continues. This hides bugs.

**Fix:** Catch specific exceptions (`ValueError`, `pd.errors.ParserError`), log with
`logger.exception()`, and let the caller decide.

> **Status:** ⬜ Not started.

---

### 2.7 Use `ruff` for linting and formatting

- Add `ruff` to dev dependencies
- Add `[tool.ruff]` config to `pyproject.toml` with rules: `E`, `F`, `I`, `UP`
  (pyupgrade), `B` (bugbear)

This catches: unused imports, f-string conversion, old-style `%` formatting, mutable
defaults.

> **Status:** ⬜ Not started.

---

### 2.8 Reduce `hysteresis.py` (37 KB) complexity

**Split into:**
- `processing/hysteresis/analysis.py` — core loop detection algorithms
- `processing/hysteresis/plotting.py` — visualization functions
- `processing/hysteresis/outliers.py` — outlier removal specific to hysteresis

> **Status:** ⬜ Not started.

---

## 3. Extensibility Improvements

### 3.1 Introduce a `DataLoader` Protocol for all data sources

**Pattern:** Already exists in `hybrid/data_protocol.py` — generalise it.

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

Register loaders in a dict: `{"txt": PupitreLoader, "csv": CsvLoader, "tdms": TdmsLoader}`

`MagnetData.from_file(path)` auto-detects format via extension.

> **Status:** ⬜ Not started.

---

### 3.2 Format auto-detection factory method

**Add to `MagnetData`:**

```python
@classmethod
def from_file(cls, path: str, **kwargs) -> "MagnetData":
    suffix = Path(path).suffix.lower()
    loader = _LOADER_REGISTRY.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported format: {suffix}")
    return loader.load(path, **kwargs)
```

Adding a new format = creating a new loader class and registering it, no `MagnetData`
changes needed.

> **Status:** ⬜ Not started.

---

### 3.3 Plugin-style analysis algorithm registry

**Pattern:** `processing/` algorithms are scattered functions — create a registry.

**File:** `python_magnetrun/processing/registry.py`

```python
_SMOOTHERS: dict[str, Callable] = {}
_DETECTORS: dict[str, Callable] = {}

def register_smoother(name: str):
    def decorator(fn): _SMOOTHERS[name] = fn; return fn
    return decorator
```

CLI `--smoother savgol` dispatches via `_SMOOTHERS["savgol"]`.

> **Status:** ⬜ Not started.

---

### 3.4 Statistics plugin interface

**Problem:** Adding new stats requires modifying `getStats()` in `magnetdata.py`.

**Solution:** Stats functions registered via decorator:

```python
@register_stat("energy")
def compute_energy(data: pd.DataFrame) -> float: ...
```

`getStats(field, extra=["energy"])` runs registered stats automatically.

> **Status:** ⬜ Not started.

---

### 3.5 Site configuration via YAML/TOML files

**Problem:** Adding a new site (e.g., M11) requires code changes.

**Solution:** Load site configs from `~/.config/magnetrun/sites.yaml` or
package-bundled `data/sites.yaml`:

```yaml
M9:
  IH_channels: [Idcct1, Idcct2]
  IB_channels: [Idcct3, Idcct4]
  flow_mapping: {Flow1: FlowH, Flow2: FlowB}
M8:
  IH_channels: [Idcct3, Idcct4]
  IB_channels: [Idcct1, Idcct2]
  flow_mapping: {Flow1: FlowB, Flow2: FlowH}
```

`prepareData()` loads from config dict — no code changes to add new sites.

> **Status:** ⬜ Not started. This is the natural next step after item 2.4. The
> `separate-cooling` branch established the config-dict API; this item would define
> where those dicts come from (a YAML file instead of call-site literals).

---

## 4. Functionality: python_magnetapi Integration

### 4.1 Create `python_magnetrun/api/` client module

**File structure:**
```
python_magnetrun/api/
├── __init__.py
├── client.py      # MagnetAPIClient class
├── models.py      # Pydantic response models
├── auth.py        # Token/credential management
└── cli.py         # magnetrun-api CLI subcommands
```

`client.py` wraps `python_magnetapi`'s REST endpoints:

```python
class MagnetAPIClient:
    def __init__(self, base_url: str, api_key: str | None = None): ...
    def get_run(self, run_id: int) -> MagnetRunRecord: ...
    def list_runs(self, site: str = None, ...) -> list[MagnetRunRecord]: ...
    def upload_run(self, run: MagnetRun) -> int: ...
    def get_stats(self, run_id: int) -> dict: ...
```

Credentials via environment variables: `MAGNETAPI_URL`, `MAGNETAPI_KEY` (consistent
with existing `userdb.py` pattern).

> **Status:** ⬜ Not started.

---

### 4.2 `MRecord.getData()` via API

**Current:** `MRecord.getData()` directly calls `requests/connect.py`
(srv-data scraping).

**Proposed:** If `MAGNETAPI_URL` is set, use `MagnetAPIClient.get_run_data()` instead
of scraping. Fallback to scraping for backwards compatibility.

> **Status:** ⬜ Not started.

---

### 4.3 Upload processed results to magnetdb

Add `MagnetRun.upload(client: MagnetAPIClient)` method:
- Serializes `MagnetRun` + `Signature` + stats to JSON
- Posts to `/api/runs/` endpoint
- Returns assigned `run_id` for tracking

> **Status:** ⬜ Not started.

---

## 5. Functionality: Interactive Dashboards

### 5.1 Expand `panels/` into a proper dashboard module

**Current state:** Only 2 scripts (`panel-mrecord.py`, `panel-mrecord-vs-time.py`),
neither importable.

**Target structure:**
```
python_magnetrun/dashboards/
├── __init__.py
├── run_overview.py      # Time-series overview dashboard
├── field_analysis.py    # Field vs current dashboard
├── comparison.py        # Multi-run comparison
├── hybrid_monitor.py    # Real-time FEPC data view
└── widgets.py           # Shared Panel widgets
```

Use `panel` + `hvplot` (already implied by `panels/panel-mrecord.py`).

> **Status:** ⬜ Not started.

---

### 5.2 `run_overview` dashboard

- Fields: field strength, currents (IH, IB), water flow, temperatures vs. time
- Controls: time range slider, field selector checkboxes, smoothing toggle
- Regime annotations: overlay U/P/D regime markers from `Signature`
- Entry point: `magnetrun-dashboard` CLI command (`panel serve ...`)

> **Status:** ⬜ Not started.

---

### 5.3 `comparison` dashboard

- Load multiple `MagnetRun` objects (from files or via API)
- Overlay field profiles, normalise time axes
- Show statistics table per run

> **Status:** ⬜ Not started.

---

### 5.4 Jupyter notebook support

- Add `magnetrun-to-notebook` CLI: generates a pre-filled `.ipynb` from a data file
- Uses `nbformat` to programmatically create notebooks with standard analysis cells

> **Status:** ⬜ Not started.

---

## 6. Additional Quick Wins

### 6.1 Fix `getInsert()` in `MagnetRun.py`

**Problem:** Lines 190–193 strip extension to get "insert" name but the logic is
incorrect (removes extension from path, not just filename).

**Fix:** `return Path(self.MagnetData.FileName).stem`

> **Status:** ⬜ Not started.

---

### 6.2 Add `__all__` to package `__init__.py`

Controls what `from python_magnetrun import *` exports; improves discoverability.

> **Status:** ⬜ Not started.

---

### 6.3 Consistent `__repr__` via dataclasses or `__str__`

`MagnetRun.__repr__` (line 179) uses old `%r` formatting instead of f-strings.

> **Status:** ⬜ Not started.

---

### 6.4 Remove module-level matplotlib side effects

**Problem:** `matplotlib.rcParams["text.usetex"] = True` at module-level import in
`python_magnetrun.py`, `outliers.py`, `pupitre.py` — breaks any non-LaTeX environment
at import time.

**Fix:** Move LaTeX config inside plot functions, guarded by an optional
`use_latex=False` parameter.

> **Status:** ⬜ Not started.

---

### 6.5 Remove hard-coded developer paths

**Problem:** `analysis/config.py` lines ~116–134 contain absolute paths like
`/home/LNCMI-G/christophe.trophime/...`

**Fix:** Use environment variables with sensible defaults: `MAGNETRUN_DATA_DIR`,
`MAGNETRUN_CONFIG_DIR`.

> **Status:** ⬜ Not started.

---

### 6.6 `bilan.py` and `flow_params.py` — document purpose

These appear to be standalone analysis scripts repurposed as modules; add module-level
docstrings.

> **Status:** ⬜ Not started. `bilan.py` imports were updated in `separate-cooling`
> (water property functions migrated to `python_magnetcooling`) but no docstring was
> added.

---

## Critical Files to Modify

| Priority | File | Change |
|---|---|---|
| High | `python_magnetrun/magnetdata.py` | Enum for Type, split into submodules, DataLoader protocol |
| High | `python_magnetrun/MagnetRun.py` | Remove `print()`→logger, centralise housing config |
| High | `python_magnetrun/python_magnetrun.py` | Split into `commands/` submodules |
| High | `python_magnetrun/requests/` → `fetchers/` | Rename module |
| Medium | `python_magnetrun/processing/hysteresis.py` | Split into 3 sub-files |
| Medium | `python_magnetrun/deserialize.py` | Replace with dataclasses/pydantic |
| Medium | `python_magnetrun/panels/` → `dashboards/` | Restructure as importable module |
| Low | `python_magnetrun/MRecord.py` | Remove commented code, pathlib |
| New | `python_magnetrun/api/` | New API client module |
| New | `python_magnetrun/dashboards/` | New dashboard module |
| New | `data/sites.yaml` | Externalised site configuration |
| New | `python_magnetrun/protocols.py` | DataLoader protocol |

---

## Recommended Libraries to Adopt

| Purpose | Library | Already used? |
|---|---|---|
| Data validation / serialization | pydantic v2 | No |
| Linting + formatting | ruff | No |
| Type checking | mypy | No |
| Dashboard framework | panel + hvplot | Partially (`panels/`) |
| Config files | `tomllib` (stdlib 3.11+) or pyyaml | No |
| CLI framework | typer (optional upgrade from argparse) | No |
| Notebook generation | nbformat | No |
| REST API client | httpx (async-capable) | No (use over requests) |
| Property-based testing | hypothesis | No |

---

## Migration Phases

| Phase | Scope |
|---|---|
| 1 | Enum refactor, print→logger, `protocols.py` at package root, add mypy+ruff, write unit tests for core |
| 2 | Split `magnetdata.py`, rename `requests/`→`fetchers/`, rename `python_magnetrun.py`→`cli_main.py`, split `hysteresis.py`, format registration system |
| 3 | Batch reporting, Jupyter `_repr_html_`, pydantic serialization, `api_client.py`, `db_adapter.py`, centralised `config.py` |
| 4 | Dashboards, streaming monitor skeleton, remove all deprecated shims |

---

## Verification Plan

1. **Unit tests:** `pytest tests/` — all existing tests pass; new tests for core modules added.
2. **Import test:** `python -c "import python_magnetrun; from python_magnetrun import MagnetData, MagnetRun, MRecord"` succeeds.
3. **CLI smoke tests:**
   - `python-magnetrun --help` shows all subcommands
   - `python-magnetrun <datafile.txt> info --list` lists keys
   - `python-magnetrun <datafile.txt> stats` shows statistics
4. **Backwards compat:** Existing example scripts in `examples/` still run without modification.
5. **Dashboard:** `magnetrun-dashboard <datafile.txt>` opens browser panel at localhost.
6. **API client:** `from python_magnetrun.api import MagnetAPIClient` imports cleanly; can be instantiated with a mock URL.
