# Code Review — python_magnetrun

*Reviewed: 2026-03-23*

---

## Package Overview

~35K lines across 91 Python files. A data analysis toolkit for magnet experimental runs,
handling TDMS/CSV/TXT acquisition files, cooling circuit analysis, and signal processing.

**5 CLI entry points:**
- `python-magnetrun` → `python_magnetrun.python_magnetrun:main`
- `magnetrun-analysis` → `python_magnetrun.analysis:main`
- `hybrid-magnetrun` → `python_magnetrun.hybrid.cli:main`
- `srvdata-to-magnetrun` → `python_magnetrun.requests.cli:main`
- `magnetrun-alimconfig` → `python_magnetrun.configAlims.convertxml:main`
- `magnetrun-pigbrother-logparser` → `python_magnetrun.tdms.log_parser:main`

---

## Directory Structure

```
python_magnetrun/
├── __init__.py
├── MagnetRun.py              (~350 lines) — high-level wrapper around MagnetData
├── magnetdata.py             (~1500 lines) — unified file I/O and data abstraction
├── python_magnetrun.py       (~1300 lines) — CLI entry point + business logic
├── signature.py              — regime/signature extraction
├── waterflow_pipeline.py     — hydraulic parameters for cooling circuit analysis
├── outliers.py               — outlier detection strategies
├── analysis/                 — comprehensive time-series analysis framework
│   ├── cli.py                — logging, progress tracking
│   ├── config.py             — site/channel configuration (M8, M9, M10)
│   ├── loaders.py            — data loading from TDMS/txt/csv
│   ├── synchronization.py    — time alignment utilities
│   ├── metrics.py            — DTW, correlation, distance metrics
│   ├── plotting.py           — multi-source visualization (915 lines)
│   └── processing.py         — core data processing pipeline
├── requests/                 — database/web interface
│   ├── cli.py
│   ├── GObject.py            — magnet component object
│   ├── HMagnet.py            — hybrid magnet wrapper
│   ├── MRecord.py            — magnet record
│   ├── connect.py            — database connection
│   ├── deserialize.py        — JSON/XML parsing
│   └── webscrapping.py       — web data extraction
├── hybrid/                   — FEPC high-frequency acquisition systems
│   ├── hybrid_data.py        — low-level reader (kHz/RMS/trigger), lazy loading
│   ├── hybrid_run.py         — MagnetRun-compatible interface (~920 lines)
│   ├── plotting.py           — hybrid-specific visualization
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
│   ├── convert.py            — unit conversion, timestamps
│   ├── plots.py              — basic plotting helpers
│   ├── list.py               — list utilities
│   ├── sequence.py           — sequence processing
│   └── duplicates.py         — duplicate detection
├── tdms/
│   └── log_parser.py         — LabVIEW/pigbrother log parsing
├── configAlims/              — configuration management
└── panels/                   — panel/dashboard plotting (underdeveloped)
```

---

## Code Statistics

| Metric | Value |
|--------|-------|
| Total lines of code | ~35K |
| Python files | 91 |
| Classes | 79 |
| Functions | 701 |
| Functions with type hints | ~284 (40%) |
| Docstrings | 1190+ |
| Try/except blocks | 392 |
| `print()` statements | 1466 |
| `logging` calls | 124 |
| TODO/FIXME/BUG comments | 46 |

---

## Strengths

- **Comprehensive documentation** — 1190+ docstrings with parameter descriptions
- **Good architecture** — clear separation into `analysis/`, `processing/`, `requests/`, `hybrid/`, `utils/`
- **Factory methods** on `MagnetData` / `MagnetRun` for different file formats (`fromtdms`, `fromtxt`, `fromcsv`, `fromStringIO`)
- **Modern Python** in newer modules — dataclasses, `@property`, context managers (`LogContext`, `timed_operation`)
- **Strategy pattern** for outlier detection (IQR, LOF, Isolation Forest)
- **Active error handling** — 392 try/except blocks
- **`analysis/` module is exemplary** — well-typed, documented, focused

---

## Issues

### High Priority

#### 1. Bare `except:` clauses (5 instances)

Files: `utils/plots.py`, `utils/txt2csv.py`, `requests/cli.py`,
`hybrid/trigger/plot_trigger_data.py`, `hybrid/vprocess/test.py`

Catches `SystemExit` and `KeyboardInterrupt`, silently swallows real errors.
Replace with specific exception types (`ValueError`, `OSError`, etc.).

#### 2. `print()` vs `logging` imbalance

1466 `print()` calls vs 124 `logging` calls. Debug output is not suppressible
when the package is used as a library. Standardize on `logger.debug()` / `logger.info()`.

#### 3. Hollow test suite

`tests/test_python_magnetrun.py` has **0 assertions**. Most `processing/` modules
have no unit tests at all. Target 70%+ coverage for core modules.

#### 4. Incomplete type hints

Only ~40% of functions have return type hints. Inconsistent use of `|` (PEP 604)
vs `Optional[X]`. Target 100% for new/modified code.

#### 5. Timestamp parsing duplication

Similar timestamp parsing logic repeated in 3+ locations. Extract to a single
utility in `utils/convert.py`.

### Medium Priority

#### 6. `magnetdata.py` — 1500-line monolith

~50 methods mixing file I/O, data transformation, and unit handling.
Split into focused submodules: `magnetdata/io.py`, `magnetdata/transform.py`, `magnetdata/query.py`.

#### 7. `python_magnetrun.py` — 1300-line CLI/logic mix

CLI argument parsing and business logic are interleaved.
Extract business logic to `commands/` submodules; keep `cli.py` thin.

#### 8. Legacy/active code coexistence

`prepareData_legacy()` and `prepareData()` coexist in `MagnetRun.py` with no
clear deprecation path. Remove or formally deprecate.

#### 9. Magic numbers scattered throughout

Energy balance thresholds, default flow parameters, sampling rates hardcoded
in multiple files. Centralize in `analysis/config.py`.

#### 10. No input validation at file boundaries

Extension and format not checked before parsing in several loaders.
Add early validation in `analysis/loaders.py` and `magnetdata.py`.

### Low Priority

- `requests/cli.py` contains a placeholder ("Replace this message...") still in place
- File paths use string concatenation instead of `pathlib.Path`
- No lock file (`uv.lock`) for reproducible installs
- No pre-commit hooks for `ruff` / `mypy` (both listed as dev deps but not configured)
- `panels/` directory has only 2 minimal examples; no multi-source dashboard support

---

## Data Sources & Interfaces

| Source | Class | Factory | `getData()` return | Sampling |
|--------|-------|---------|-------------------|----------|
| Pigbrother Overview | `MagnetData` (Type=0) | `fromtdms()` | `DataFrame` | 1 Hz |
| Pigbrother Archive | `MagnetData` (Type=1) | `fromtdms()` | `DataFrame` | 120 Hz |
| Pupitre | `MagnetData` (Type=0) | `fromtxt()` | `DataFrame` | ~1 Hz |
| Hybrid kHz/RMS | `HybridData` | constructor | `(array, time)` tuple | 1 kHz / variable |

**Key mismatch**: TDMS/Pupitre return DataFrames; Hybrid returns `(array, time)` tuples.
A `DataProvider` protocol is defined in `hybrid/hybrid_run.py` but not enforced.

---

## Current Plotting Capabilities

| Capability | TDMS Overview/Archive | Pupitre | Hybrid kHz/RMS |
|------------|----------------------|---------|----------------|
| Individual plotting | `MagnetData.plotData()` | `MagnetData.plotData()` | `hybrid/plotting.py` |
| Multi-variable | `utils/plots.py` | `utils/plots.py` | `hybrid/plotting.py` |
| Same-axes overlay | `analysis/plotting.plot_data()` | `analysis/plotting.plot_data()` | **Not supported** |
| Time alignment | Manual | Manual | Manual |
| Channel mapping | Manual (`channels_dict`) | Manual (`pupitre_dict`) | Manual |

`analysis/plotting.plot_data()` already overlays Overview + Archive + Pupitre
on shared axes — hybrid integration is the main gap.

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
| Factory methods | `MagnetData.fromtdms/fromtxt`, `MagnetRun.fromtdms/fromtxt`, `HybridRun.fromdir` |
| Properties | `Signature` class |
| Dataclasses | `HydraulicData`, `SyncResult`, `DistanceResult`, `PlotStyle`, `PlotColors` |
| Context managers | `LogContext`, `timed_operation()` in `analysis/cli.py` |
| Strategy | Outlier detection (IQR, LOF, Isolation Forest) |
| Protocol (informal) | `DataProvider` in `hybrid/hybrid_run.py` |

---

## Summary

**python_magnetrun** is a functionally mature package for magnet experimental run analysis.
The `analysis/` module shows the target quality level for the rest of the codebase.
Main pain points are: consistency (print vs logging, typed vs untyped, new vs legacy patterns),
test coverage, and lack of unified plotting across all three data sources.
The "Pre-Alpha" label is appropriate — core workflows are production-ready,
but the codebase needs consistency work before a 1.0 release.
