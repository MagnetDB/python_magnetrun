# Code Review — python_magnetrun

*Reviewed: 2026-03-23 — Updated: 2026-04-30*

---

## Package Overview

~36K lines across 113 Python files. A data analysis toolkit for magnet experimental runs,
handling TDMS/CSV/TXT acquisition files, cooling circuit analysis, and signal processing.

**5 CLI entry points:**
- `python-magnetrun` → `python_magnetrun.cli:main` *(was `python_magnetrun:main`)*
- `magnetrun-analysis` → `python_magnetrun.analysis:main`
- `hybrid-magnetrun` → `python_magnetrun.hybrid.cli:main`
- `srvdata-to-magnetrun` → `python_magnetrun.requests.cli:main`
- `magnetrun-alimconfig` → `python_magnetrun.configAlims.convertxml:main`

---

## Directory Structure

```
python_magnetrun/
├── __init__.py
├── MagnetRun.py              — high-level wrapper (satisfies DataLoader protocol)
├── magnetdata.py             — backward-compat facade (~235 lines)
├── magnetdata_base.py        — ABC: MagnetDataBase (Data as abstract property; close()/context manager)
├── magnetdata_pandas.py      — PandasMagnetData + subclasses; lazy load via _ensure_data_loaded()
├── magnetdata_tdms.py        — TdmsMagnetData; lazy group load via _LazyGroupDict
├── cli.py                    — CLI entry point (was python_magnetrun.py)
├── housing_config.py         — HousingConfig: single source of truth for site/sensor config
├── log_utils.py              — structured logging utilities
├── field_defs.py             — field definitions
├── signature.py              — regime/signature extraction
├── waterflow_pipeline.py     — hydraulic parameters for cooling circuit analysis
├── bfield/                   — NEW: B-field profile data adapter
│   └── bfield_run.py         — BFieldRun (DataLoader protocol)
├── simulation/               — NEW: simulation data adapters
│   ├── simulation_run.py     — SimulationRun (DataLoader protocol)
│   └── magnettools_reader.py — magnettools file reader (stub)
├── plotting/                 — NEW: backend-agnostic plotting subpackage
│   ├── backend.py            — PlottingBackend protocol
│   ├── matplotlib_backend.py — Matplotlib implementation
│   ├── plotly_backend.py     — Plotly implementation
│   ├── plotly_resampler_backend.py — Plotly-Resampler implementation
│   ├── timeseries.py         — time-series plotting utilities
│   ├── style.py              — plot styling
│   └── annotations.py        — annotations support
├── analysis/                 — comprehensive time-series analysis framework
│   ├── cli.py                — CLI with logging, progress tracking (589 lines)
│   ├── config.py             — channel/site configuration
│   ├── loaders.py            — data loading (1403 lines; UnicodeDecodeError handling added)
│   ├── synchronization.py    — time alignment utilities
│   ├── metrics.py            — DTW, correlation, distance metrics
│   ├── plotting.py           — multi-source visualization (926 lines)
│   └── processing.py         — core data processing pipeline (1293 lines)
├── hybrid/                   — FEPC high-frequency acquisition systems
│   ├── hybrid_data.py        — low-level reader (kHz/RMS/trigger), lazy loading
│   ├── hybrid_run.py         — satisfies DataLoader protocol; hybrid_formula_map key resolution
│   ├── data_protocol.py      — DataLoader protocol definition
│   ├── kHz/                  — 1 kHz data handling
│   ├── rms/                  — RMS data handling
│   ├── trigger/              — trigger events
│   └── vprocess/             — voltage processing
├── processing/               — signal/time-series processing
│   ├── smoothers.py          — Savitzky-Golay, LOESS
│   ├── trends.py             — trend detection
│   ├── correlations.py       — lag/cross-correlation
│   ├── peaks.py              — peak detection
│   ├── breakingpoints.py     — piecewise fitting
│   ├── fit.py                — curve fitting
│   ├── hysteresis.py         — hysteresis loops
│   ├── stats.py              — statistical functions
│   └── distance.py           — similarity metrics
├── utils/
│   ├── files.py              — file expansion, data loading
│   ├── downsampling.py       — NEW: DownsampleConfig + shared downsampling utilities
│   ├── timestamps.py         — NEW: timestamp parsing utilities
│   ├── timezone.py           — NEW: timezone helpers
│   ├── validation.py         — FileFormatError + validators; check_pupitre_truncation()
│   ├── txt2csv.py            — text to CSV conversion
│   ├── plots.py              — basic plotting helpers
│   ├── list.py               — list utilities
│   ├── sequence.py           — sequence processing
│   └── duplicates.py         — duplicate detection
├── requests/                 — database/web interface
├── tdms/                     — LabVIEW/pigbrother log parsing
├── configAlims/              — configuration management
└── panels/                   — panel/dashboard plotting (underdeveloped)
```

---

## Code Statistics

| Metric | Mar 2026 | Apr 2026 | Apr 2026 (latest) | Trend |
|--------|----------|----------|--------------------|-------|
| Total lines of code | ~35K | ~36K | ~37K | +1K |
| Python files (package) | 91 | 113 | 111 | → stable |
| `logger.*` calls | 124 | 1140 | 1191 | ✅ +5% |
| `print()` (diagnostic) | 1466 | ~13 | ~13 | ✅ low |
| `print()` (intentional CLI output) | — | — | ~430 | → acceptable |
| Bare `except:` clauses | 5 | 0 | 0 | ✅ Fixed |
| TODO/FIXME/BUG comments | 46 | 34 | ~34 | → stable |
| Functions with type hints | ~284 (40%) | ~40% | ~40% | → Ongoing |

*Note: `print()` count includes intentional console output in housing_config display, `commands/stats.py` progress lines, and other CLI-output paths. Diagnostic `print()` (non-CLI) remains near zero.*

---

## Strengths

- **Comprehensive documentation** — docstrings with parameter descriptions throughout
- **Good architecture** — clear separation into `analysis/`, `processing/`, `requests/`, `hybrid/`, `utils/`, `plotting/`
- **Factory methods** on `MagnetData` / `MagnetRun` for different file formats (`fromtdms`, `fromtxt`, `fromcsv`, `fromStringIO`)
- **Modern Python** in newer modules — dataclasses, `@property`, context managers (`LogContext`, `timed_operation`)
- **Strategy pattern** for outlier detection (IQR, LOF, Isolation Forest)
- **Unified DataLoader protocol** — `MagnetRun`, `HybridRun`, `SimulationRun`, `BFieldRun` all satisfy one protocol
- **Logging infrastructure** — `log_utils.py`, diagnostic print→logger migration essentially complete
- **File validation** — `utils/validation.py` with `FileFormatError` + `check_pupitre_truncation()` integrated throughout loaders
- **`ruff` pre-commit hook** — enforces style consistency on every commit
- **Lazy loading** — `PandasMagnetData` defers full CSV load to first `Data` access; `TdmsMagnetData` defers per-group TDMS load via `_LazyGroupDict`
- **`Data` as abstract property** — lazy-load contract part of the ABC; `close()` + context-manager on base class enables `with load_magnetdata(...) as m:`
- **Resilient file loading** — encoding fallback (UTF-8 → Latin-1), `on_bad_lines="warn"`, empty-data guard; `UnicodeDecodeError` caught in all loaders
- **Rich field metadata** — `addData`/`computeData` now carry `symbol`, `unit`, `label`, `description`; stored as `FieldMeta` for automatic axis labelling
- **Formula-key resolution in `HybridRun`** — `hybrid_formula_map` entries (e.g. `FEPC-AUX-LNCMI/ALIM1`) resolved by `_resolve_hybrid_formula` instead of failing with `ValueError`

---

## Issues

### High Priority

#### 1. ~~Bare `except:` clauses~~ ✅ FIXED

All 5 bare except clauses removed. Replaced with specific exception types.

#### 2. ~~`print()` vs `logging` imbalance~~ ✅ MOSTLY FIXED

Down from 1466 `print()` to 13 remaining; 1140 structured `logging` calls.
Remaining 13 prints are in non-critical paths — low priority cleanup.

#### 3. Hollow test suite 🔴 OPEN (improving)

`tests/test_python_magnetrun.py` still has **0 assertions**. `tests/analysis/` has 7
test files. New additions: `test_truncated_pupitre.py` (6 cases), `test_hybrid_formula_resolution.py`
(5 cases, mocked), `test-vprocess.py` (496 lines), `test-cfg-parser.py` (135 lines).
Most `processing/` modules still lack unit tests. Target 70%+ coverage for core modules.

#### 4. Incomplete type hints 🔴 OPEN

Still ~40% coverage. Inconsistent use of `|` (PEP 604) vs `Optional[X]`.
Target 100% for new/modified code; enable `mypy` in CI when coverage improves.

#### 5. ~~Timestamp parsing duplication~~ ✅ FIXED

Extracted to `utils/timestamps.py` and `utils/timezone.py`.

#### 6. Multiple-file `vs_time` regression 🔴 OPEN

Plot timing issues when multiple input files are used (commits 86c45c6/76351f3).
Root cause not yet identified.

#### 7. kHz/RMS timestamp not UTC 🔴 OPEN

kHz/RMS data uses seconds-from-day-start rather than UTC timestamps.
Blocks Phase 2B (time alignment) and multi-source plotting.

### Medium Priority

#### 8. ~~`magnetdata.py` — 1500-line monolith~~ ✅ FIXED

Split into `magnetdata_base.py` (ABC), `magnetdata_pandas.py`, `magnetdata_tdms.py`.
`magnetdata.py` is now a ~235-line backward-compat facade.

#### 9. `cli.py` — still needs further splitting 🟡 PARTIAL

`python_magnetrun.py` was renamed to `cli.py` and args extracted. Business logic
body still needs splitting into `commands/` submodules.

#### 10. ~~Legacy/active code coexistence~~ ✅ FIXED

`prepareData_legacy()` removed. `runetl.prepareData` now fully driven by `HousingConfig`.

#### 11. Magic numbers scattered throughout 🔴 OPEN

Energy balance thresholds, default flow parameters, sampling rates hardcoded
in multiple files. Centralize in `analysis/config.py`.

#### 12. ~~No input validation at file boundaries~~ ✅ FIXED

`utils/validation.py` with `FileFormatError` integrated in all loaders.

#### 13. ~~`addData`/`computeData` lacked metadata~~ ✅ FIXED

Both methods now accept `symbol`, `unit`, `label`, `description` and store `FieldMeta` in
`self.field_meta[key]`.  Housing-config formula maps and JSON definition files carry the
same four keys.  Plot label propagation is now automatic.

#### 14. ~~`HybridRun.getData` failed on `hybrid_formula_map` keys~~ ✅ FIXED

`_resolve_hybrid_formula()` helper evaluates formula strings (`LHS = op1 + op2 + …`) by
loading constituent `kHz/…` channels and returning their element-wise sum.  Result is
cached transparently.

### Low Priority

- `requests/cli.py` contains a placeholder ("Replace this message...") still in place
- File paths: ~90% migrated to `pathlib.Path`; some `os.path` usage remains
- No lock file (`uv.lock`) for reproducible installs
- No CI/CD pipeline yet (`.github/workflows/ci.yml` not created)
- `mypy` pre-commit hook exists but not enabled
- `panels/` directory has only 2 minimal examples; no multi-source dashboard support

---

## Data Sources & Interfaces

| Source | Wrapper | Factory | DataLoader? | Sampling |
|--------|---------|---------|-------------|----------|
| Pigbrother Overview/Archive | `MagnetRun` | `load_magnetdata()` | ✅ | 1 Hz / 120 Hz |
| Pupitre | `MagnetRun` | `load_magnetdata()` | ✅ | ~1 Hz |
| Hybrid kHz/RMS | `HybridRun` | `HybridRun.fromdir()` | ✅ | 1 kHz / variable |
| Ensight / Feel++ simulation | `SimulationRun` | `from_ensight()` / `from_feelpp()` | ✅ | spatial / transient |
| Magnetic field profile | `BFieldRun` | `from_bprofile()` | ✅ | spatial |

**Protocol:** All sources satisfy `DataLoader` (defined in `hybrid/data_protocol.py`).
`get_time_range()` and `getDomain()` are part of the protocol.

**Remaining gap:** kHz/RMS `get_time_range()` uses seconds-from-day-start internally;
needs UTC conversion before `align_to_common_time()` can be implemented.

---

## Current Plotting Capabilities

| Capability | TDMS Overview/Archive | Pupitre | Hybrid kHz/RMS |
|------------|----------------------|---------|----------------|
| Individual plotting | `MagnetRun` + `plotting/` backends | `MagnetRun` + `plotting/` backends | `hybrid/plotting.py` |
| Multi-variable | ✅ via `plotting/timeseries.py` | ✅ | ✅ |
| Same-axes overlay | `analysis/plotting.plot_data()` | `analysis/plotting.plot_data()` | 🔴 Not yet |
| Backend choice | matplotlib / plotly / plotly-resampler | same | separate |
| Time alignment | UTC via `PandasMagnetData.addTime()` | UTC via `addTime()` | 🔴 Not UTC yet |
| Channel mapping | `HousingConfig` | `HousingConfig` | manual |
| Downsampling | `DownsampleConfig` | `DownsampleConfig` | `DownsampleConfig` |

**Plotting backends** (`plotting/`): `PlottingBackend` protocol with 3 implementations —
`MatplotlibBackend`, `PlotlyBackend`, `PlotlyResamplerBackend`. Field-style support added.

**Main gap:** Hybrid kHz/RMS not yet on shared axes with pupitre/TDMS (Phase 2C, depends on UTC timestamps in 2B).

---

## Dependencies

**Core:** pandas, numpy, scipy, matplotlib, seaborn, lxml, iapws, pint, nptdms, natsort, nlopt, pytz, python_magnetcooling

**Optional `signal` group:** scikit-learn, stumpy, ruptures, pwlf, sympy, pyextremes, xmltodict

**Optional `system` group:** clawpack, ht

**Dev:** pytest, ruff

---

## Design Patterns in Use

| Pattern | Where |
|---------|-------|
| Factory function | `load_magnetdata()` in `magnetdata.py` — replaces shim class |
| Factory methods | `MagnetRun.fromtdms/fromtxt`, `HybridRun.fromdir`, `SimulationRun.from_*`, `BFieldRun.from_*` |
| Protocol (formal) | `DataLoader` in `hybrid/data_protocol.py` — satisfied by all 4 run wrappers |
| Protocol (formal) | `PlottingBackend` in `plotting/backend.py` — 3 implementations |
| Dataclasses | `HydraulicData`, `SyncResult`, `DistanceResult`, `PlotStyle`, `PlotColors`, `DownsampleConfig` |
| Context managers | `LogContext`, `timed_operation()` in `analysis/cli.py`; `MagnetDataBase.__enter__`/`__exit__` for file-handle cleanup |
| Strategy | Outlier detection (IQR, LOF, Isolation Forest) |
| Single source of truth | `HousingConfig` for site/sensor/channel configuration |
| Abstract property | `MagnetDataBase.Data` — lazy-load contract is part of the ABC |
| Named tuple | `FieldMeta(symbol, unit, label, description)` — propagated by `addData`/`computeData` |

---

## Summary

**python_magnetrun** is a production-ready package for magnet experimental run analysis.
Significant progress since March 2026: logging migration is essentially complete for
diagnostic paths, the magnetdata monolith has been split, `Data` is now an abstract
property with lazy loading and context-manager support on the ABC, a unified `DataLoader`
protocol now covers all data sources (including `SimulationRun` and `BFieldRun` adapters),
plotting has been refactored into a multi-backend subpackage, file loading is resilient to
encoding and truncation issues, `addData`/`computeData` carry full `FieldMeta`, and
`HybridRun.getData` correctly resolves `hybrid_formula_map` keys.

**Main remaining pain points:**
1. kHz/RMS UTC timestamp conversion (blocks unified multi-source plotting, Phase 2B)
2. Multiple-file `vs_time` regression
3. Type hint coverage still ~40%
4. No CI/CD pipeline

The "Pre-Alpha" label has been effectively outgrown for core workflows.
A 1.0 release is within reach once the multi-source plotting roadmap (Phases 2B–2D)
and CI/CD are complete.
