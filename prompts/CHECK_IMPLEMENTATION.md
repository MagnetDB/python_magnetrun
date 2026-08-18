# Implementation Status — python_magnetrun

*Last updated: 2026-06-09 — branch `rework_analysis`*

This document tracks detailed implementation status and task completion. For strategic direction, see [ROADMAP.md](ROADMAP.md). For architectural review, see [REVIEW.md](REVIEW.md).

---

## Package Status: Production-Ready ✅

The package is stable and functional for core use cases. All critical structural issues have been resolved. Focus is now on stability improvements, unified plotting features, and internal refactoring.

---

## Package Structure

```
python_magnetrun/
├── magnetdata_base.py       # ABC (DataType: PUPITRE/TDMS/ENSIGHT/HYBRID/HTS)
├── magnetdata_pandas.py     # Pandas impl — factory methods delegate to readers/
├── magnetdata_tdms.py       # TDMS impl
├── magnetdata.py            # Factory entry point (load_magnetdata accepts fmt= override)
├── MagnetRun.py             # Session container
├── runetl.py                # ETL helpers
├── outliers.py              # Canonical outlier detection (OutlierConfig, OutlierDetector, OUTLIER_DEFAULTS)
├── field_defs.py / housing_config.py  # Config layer
├── feelpp-defs.json         # (planned Phase H2) pattern-based defs for feelpp/paraview
├── cli.py                   # CLI entry point (renamed from python_magnetrun.py)
├── cli_args.py / args.py    # CLI argument parsing (create_outlier_parser, args_to_outlier_config)
├── commands/                # Modular CLI subcommands
├── readers/                 # Pure I/O readers — one class per format (Stream 3.6 ✅)
│   ├── base.py              #   Reader protocol (runtime-checkable)
│   ├── csv_readers.py       #   PupitreReader, BProfileReader, EnsightReader, FeelppReader, CsvReader
│   ├── tdms_reader.py       #   TdmsReader (validate + t-offset config)
│   ├── hts_reader.py        #   HtsReader (new: ; sep, units-in-header, extracted_units())
│   ├── hybrid_reader.py     #   HybridReader (composite discovery)
│   └── registry.py          #   READERS/CONTAINERS + detect_type()
├── analysis/                # Analysis pipeline
├── hybrid/                  # FEPC kHz/RMS/Trigger data (outliers.py is a backward-compat shim)
│   └── hybrid_data.py       #   HybridData now inherits MagnetDataBase (Stream 3.6 R4 ✅)
├── processing/              # Signal processing (signal.py: normalize_signal, binarize_signal, _otsu_threshold)
├── plotting/                # Plotting backends & utilities
├── simulation/              # SimulationRun adapter (Phase B done)
├── bfield/                  # BFieldRun adapter (Phase C done)
├── comparison/              # (planned Phases D-G) ComparisonSession, KeyMapping, CLI
├── utils/ / runlogs/ / requests/ / configAlims/
```

**Class Hierarchy:**

```
MagnetDataBase (ABC)
├── PandasMagnetData
│   ├── EnsightMagnetData
│   ├── BProfileMagnetData
│   └── FeelppMagnetData
├── TdmsMagnetData
└── HybridData              ← joined hierarchy (Stream 3.6 R4 ✅); field_meta init bug fixed

readers/ subpackage         ← pure I/O; factory methods delegate to readers (Stream 3.6 ✅)

load_magnetdata(filename, fmt=)  ← uses detect_type() from registry (magnetdata.py)

MagnetRun                   ← owns a MagnetDataBase instance, uses load_magnetdata
HybridRun                   ← satisfies DataLoader protocol
```

---

## Work Streams Status

### Stream 1: Production Stability (High Priority)

#### Known Issues & Blockers

| Issue | Status | Effort | Priority | Notes |
|-------|--------|--------|----------|-------|
| Multiple-file `vs_time` regression | 🔴 Open | 1-2 days | **Critical** | Plot timing issues (commits 86c45c6/76351f3) |
| Enable `mypy` pre-commit / CI | 🔴 Open | 30 min | **Medium** | `ruff` already enforced via pre-commit (`--fix`); `mypy` hook exists but is commented out |
| Validation on all entry points | 🟡 Verify | 2-3 hours | **High** | Infrastructure exists; verify all paths use it |
| Logging migration | 🟡 In Progress | Ongoing | **Medium** | ~100-200 `print()` calls remain |

#### Quality & Validation ✅ Mostly Complete

| Task | Status | Commit | Notes |
|------|--------|--------|-------|
| Fix bare `except:` clauses | ✅ Done | `6253de6` | No bare excepts remain in key files |
| File format validation | ✅ Done | `3e722ec` | `utils/validation.py` integrated throughout |
| `ruff` pre-commit hook | ✅ Done | `64ea699` | Enforces consistency |
| `pathlib.Path` migration | 🟡 Partial | — | ~90 occurrences converted; ongoing |
| Truncated pupitre file handling | ✅ Done | `8c4da77` | Encoding fallback, `on_bad_lines`, `check_pupitre_truncation`, `UnicodeDecodeError` in callers |

#### Test Infrastructure ✅ Mostly Complete

| Task | Status | Details |
|------|--------|---------|
| `tests/analysis/` suite | ✅ Done | 7 test files: config, loaders, metrics, plotting, processing, sync, CLI |
| `tests/test_file_validation.py` | ✅ Done | 261 lines, 10 test classes, 34 test methods covering all validators |
| Unit tests for `magnetdata.py` | ✅ Done | Covers factory, fromtdms, fromtxt, getData, column renaming |
| Unit tests for `processing/` | ✅ Done | Pure functions: smoothers, trends, peaks, stats |
| CLI entry point smoke tests | ✅ Done | Integration tests verify clean exits |
| `tests/test_truncated_pupitre.py` | ✅ Done | 153 lines; 6 test cases for truncation/encoding/header-only |
| `tests/test_hybrid_formula_resolution.py` | ✅ Done | 5 test cases for `HybridRun.getData` formula keys (mocked) |
| `tests/test-vprocess.py` | ✅ Done | 496 lines; vprocess reader integration tests |
| `tests/test-cfg-parser.py` | ✅ Done | 135 lines; config parser tests |
| `test_python_magnetrun.py` assertions | ✅ Done | 7 assertions: `__version__`, `__author__`, `__email__`, `load_magnetdata`, `HousingConfig`, `MagnetRun` |
| CI pipeline (GitHub Actions) | ✅ Done | `test.yml` (pytest, Ubuntu 3.11–3.14 + Debian Trixie, Codecov) + `docs.yml` already in `.github/workflows/` |

---

### Stream 2: Unified Multi-Source Plotting

#### Phase 2A: Unified Data Interface ✅ COMPLETE

| Task | Status | Location | Notes |
|------|--------|----------|-------|
| `DataLoader` protocol defined | ✅ Done | `hybrid/data_protocol.py` | Single protocol, includes `get_time_range()`, `getDomain()` |
| `DataProvider` duplication removed | ✅ Done | — | Removed from `hybrid_run.py` |
| `MagnetRun` satisfies protocol | ✅ Done | `MagnetRun.py` | Methods: `get_time_range()`, `getDomain()` |
| `HybridRun` satisfies protocol | ✅ Done | `hybrid/hybrid_run.py` | Cross-domain Phase A0–A3 complete (commit `de9f374`) |
| Protocol compliance tests | ✅ Done | `tests/test_protocol.py` | Verifies both classes satisfy protocol |

#### Phase 2B: Time Alignment Layer ✅ COMPLETE

**See:** [phase2b-time-alignment.plan.md](phase2b-time-alignment.plan.md) for full analysis and timezone bug details.

| Task | Status | Commit | Notes |
|------|--------|--------|-------|
| TDMS `wf_start_time` exposed | ✅ Done | prior | `get_time_range()` via protocol |
| Pupitre naive UTC timestamps | ✅ Done | prior | `PandasMagnetData.addTime()` converts local → UTC |
| B0.5 — Standardise `hours` = UTC in `read_khz_variable` / `read_rms_variable` | ✅ Done | `ca4b41a` | `utc_hour_to_local()` in `hybrid/utils.py`; `_utc_hour_to_local` closure removed from `analysis/processing.py`; `test_hybrid_api.py` +75 lines |
| B1 — Fix `HybridRun.get_time_range()` | ✅ Done | `e397101` | `_khz_first_last_utc(hdata)` helper in `hybrid_data.py`; returns `(t_start, t_end)` from bin-file UTC hours |
| B2 — Fix RMS time origin | ✅ Done | `e397101` | `time` now seconds since UTC midnight of recording date; `test_hybrid_api.py` +59 lines |
| B2.5 — Fix `plot_rms_variable` double-read bug | ✅ Done | prior | Uses stashed `orig_data`/`orig_time` in both highlight branches |
| B3 — `align_to_common_time(sources, reference, hours)` | ✅ Done | `e67233c`, `fb79656` | In `utils/timestamps.py`; `hours` param added; `test-timestamp.py` +55 lines; `test_hybrid_api.py` +30 lines |
| B4 — Refactor demonstrator | ✅ Done | `1ab50de` | `examples/plot_hybrid_with_pupitre_tdms.py` refactored to use `align_to_common_time()` |

#### Phase 2C: Extend `plot_data()` for Hybrid 🔴 PLANNED

**Goal:** Add hybrid data support to `analysis/plotting.plot_data()`

**Design:**
```python
def plot_data(
    ...
    df_hybrid: HybridRun | None = None,
    hybrid_channels: list[str] | None = None,
    ...
)
```

**Dependencies:** Phase 2B — ✅ **now complete**
**Effort:** ~2-3 weeks
**Status:** Unblocked — ready to start

#### Phase 2D: Side-by-Side Comparison 🔴 PLANNED

**Goal:** Multi-source comparison plots with shared time axis

**Design:** Extend `plot_comparison()` to accept `list[DataLoader]`
- Auto-generate subplot grid (source × channel)
- Linked time axis across subplots
- Consistent styling

**Dependencies:** Phase 2C
**Effort:** ~1-2 weeks

#### Phase 2E: Channel Auto-Mapping 🟡 PARTIAL

**Goal:** Automatic channel name mapping across sources

**Current State:**
- ✅ Structured `ChannelMapping` exists in `analysis/config.py`
- 🔴 Missing: `CHANNEL_ALIASES` registry
- 🔴 Missing: Fuzzy fallback for unmapped channels

**Effort:** ~1-2 weeks

---

### Stream 3: Internal Refactoring (Lower Priority)

#### 3.1 `analysis/` Subpackage Refactoring ✅ COMPLETE *(branch `rework_analysis`)*

**Status:** All 6 phases complete — 866 tests pass, 6 skipped
**See:** [analysis-subpackage-refactoring.plan.md](analysis-subpackage-refactoring.plan.md)

**Completed Phases:**
1. ✅ Dead code removed (`_extract_signatures` stubbed, `_get_archive_channel` deleted); logging migrated; `DIR_*` constants centralised in `utils/files.py`
2. ✅ Downsampling unified — `DownsampleConfig` adopted in `analysis/plotting.py`; old percent-based functions removed
3. ✅ Data loading consolidated — `utils/files.py` is canonical; `analysis/loaders.py` imports from it; `_open_text_with_fallback`, `extract_data`, `find_files`, `select_files`, `load_df`, `load_data`, `merge_data` moved
4. ✅ Channel mapping moved to `HousingConfig` — `get_pupitre_current_channel`, `get_pupitre_group_keys`, `get_pupitre_flow_keys`, `get_hybrid_group_keys` added; 5 processing.py wrappers deleted; stray debug `print()` → `logger.debug()`
5. ✅ Monolith decomposition — `FileDiscovery.discover()` (5 helpers), `process_overview_file()` (3 helpers), `analysis/cli.main()` (6 helpers including `_emit_metrics`)
6. ✅ `add_time_columns(df, t0, sampling_rate)` added to `utils/timestamps.py`; `add_time_column_with_offset` and `add_time_column` now delegate to it; inline lambdas in `load_incident_data`, `synchronize_data`, `apply_lag_correction` replaced

#### 3.2 `hybrid/` Subpackage Refactoring ✅ COMPLETE

**Status:** All 6 phases done — 866 tests pass
**See:** [hybrid-subpackage-refactoring.plan.md](hybrid-subpackage-refactoring.plan.md)

**Completed Phases:**
1. ✅ `print()` → `logger.debug/info`; commented debug code removed; `print_summary()` kept as-is
2. ✅ Outlier duplication removed from `hybrid/utils.py`; re-exports from `python_magnetrun.outliers`
3. ✅ `OUTLIER_DEFAULTS` dict centralised in `python_magnetrun/outliers.py`; `threshold or ...` bug fixed
4. ✅ `OutlierConfig` frozen dataclass (mirrors `DownsampleConfig`); `hybrid_data.py` plot methods use `outlier_config: OutlierConfig | None`; `create_outlier_parser` / `args_to_outlier_config` in `cli_args.py`; canonical outlier module moved to `python_magnetrun/outliers.py`; `hybrid/outliers.py` is a shim
5. ✅ `normalize_signal`, `binarize_signal`, `_otsu_threshold` moved to `python_magnetrun/processing/signal.py`; `processing/__init__.py` re-exports public names; shims in `hybrid/utils.py`; `hybrid_run.py` lazy import updated
6. ✅ `_evict_oldest_cache_entry()` extracted with docstring; all-NaN guard in `read_khz_variable`; file-existence guard + `FileNotFoundError` fallback in `read_rms_variable`; `load_khz_config` raises `FileNotFoundError`; `_build_groups` wraps key discovery in try/except; `saveData` guards against group key

#### 3.3 CLI Consolidation ✅ COMPLETE

**Status:** All phases done — single `magnetrun` dispatcher with 13 subcommands
**See:** [cli-consolidation.plan.md](cli-consolidation.plan.md)

**Delivered:**
- `python_magnetrun/main.py` — unified dispatcher; `register(subparsers)` pattern on all modules
- 13 subcommands: `info`, `add`, `plot`, `select`, `stats`, `signature`, `analysis`, `processing`, `hybrid`, `logparser`, `fetch`, `config` + `compare` placeholder
- `input_file` per subcommand (subcommand-first argv; `_normalize_argv` eliminated from `cli.py`)
- Old entry points kept as deprecated aliases in `pyproject.toml` for one release cycle
- `magnetrun compare` pending — blocked on `comparison/cli.py` (Phase F of cross-domain comparison)

#### 3.5 Outlier Deduplication ✅ COMPLETE

**Status:** All steps done
**See:** [outlier-consolidation.plan.md](outlier-consolidation.plan.md)

**Canonical module:** `python_magnetrun/outliers.py` — `OutlierDetector`, `OutlierResult`, `OutlierMethod` (now includes `ISOLATION_FOREST`), `OutlierConfig`, `OUTLIER_DEFAULTS`, `detect_outliers`, `remove_outliers`, `find_outlier_segments`, `get_outlier_summary`, `analyze_outliers`. `hybrid/outliers.py` is a backward-compat shim that re-exports everything.

**Completed actions:**

| File | Action | Result |
|------|--------|--------|
| `examples/outliers.py` | Deleted (`git rm`) | 213-line inline reimplementation gone |
| `processing/hysteresis.py::remove_outliers` | Replaced with thin delegator | ~120 lines → ~15 lines; delegates to `detect_outliers()` |
| `tests/test-anomalies.py` + `tests/test-anomalies-optimized.py` | Deleted (`git rm`) | CLI scripts gone |
| `tests/test_outliers.py` | Created | 142 tests (synthetic data, no file I/O); all passing |
| `hybrid/outliers.py::OutlierMethod` | `ISOLATION_FOREST` added | sklearn backend; contamination threshold; rolling rejected with `ValueError` |
| `processing/hysteresis.py::_VALID_METHODS` | `"isolation_forest"` added | Delegates through to canonical module |
| `tests/test_processing.py` | Error-message regex updated | `"method must be one of"` (new wording) |

#### 3.6 Reader/Container Split ✅ COMPLETE

**Status:** All 5 phases done — 971 tests pass (925 existing + 46 new)
**See:** [reader-container-refactoring.plan.md](reader-container-refactoring.plan.md)

| Phase | Task | Status |
|-------|------|--------|
| R1 | `PupitreReader`, `BProfileReader`, `EnsightReader`, `FeelppReader`, `CsvReader`; factory classmethods delegate | ✅ Done |
| R2 | `TdmsReader` with `validate()` + `t_offset_for()`; `required_group` on reader; `_fromtdms()` updated | ✅ Done |
| R3 | `HtsReader` (`;`-sep, `extracted_units()`); `DataType.HTS = 4` added to enum | ✅ Done |
| R4 | `HybridData(MagnetDataBase)` — `Data`/`Type` as abstract properties; `extractData`/`renameData` stubs; `getData` accepts `downsample`; `field_meta` init bug fixed; `HybridReader` composite | ✅ Done |
| R5 | `readers/registry.py` (`READERS`, `CONTAINERS`, `detect_type()`); `load_magnetdata()` uses `detect_type()`, accepts `fmt=`; `readers/__init__.py` public exports | ✅ Done |

**New files:** `readers/__init__.py`, `readers/base.py`, `readers/csv_readers.py`,
`readers/tdms_reader.py`, `readers/hts_reader.py`, `readers/hybrid_reader.py`,
`readers/registry.py`, `tests/readers/__init__.py`, `tests/readers/test_csv_readers.py`,
`tests/readers/test_tdms_reader.py`, `tests/readers/test_hts_reader.py`

#### 3.8 Downsampling Extensions ✅ COMPLETE

All three sub-phases done. See plan files for details.

**See:** [m4-downsampling.plan.md](m4-downsampling.plan.md), [rdp-downsampling.plan.md](rdp-downsampling.plan.md), [downsampling-metrics.plan.md](downsampling-metrics.plan.md)

##### 3.8a — M4 / NaN-M4

| Phase | Task | Status |
|-------|------|--------|
| 1 | Add `m4` branch to `_downsample_indices`; import `M4Downsampler` from `tsdownsample` | ✅ Done |
| 2 | Add `nan_m4` early-exit in `downsample_arrays` + `downsample_dataframe` (bypasses NaN-strip path) | ✅ Done |
| 3 | Add `'m4'`, `'nan_m4'` to `DOWNSAMPLE_METHODS` in `cli_args.py` | ✅ Done |
| 4 | `tests/test_downsampling.py` — 8 new test cases | ✅ Done |

No new dependency: `M4Downsampler` / `NaNM4Downsampler` already in `tsdownsample`.

##### 3.8b — RDP / Visvalingam-Whyatt

| Phase | Task | Status |
|-------|------|--------|
| 1 | Add `epsilon: float \| None = None` to `DownsampleConfig`; add `rdp`/`vw` dispatch + `from_n_out_rdp()` binary-search factory | ✅ Done |
| 2 | Add `rdp = ["simplification>=0.7"]` to `pyproject.toml` extras | ✅ Done |
| 3 | Add `'rdp'`, `'vw'` to `DOWNSAMPLE_METHODS`; pass `epsilon` from CLI JSON params | ✅ Done |
| 4 | `tests/test_downsampling.py` — 9 new test cases | ✅ Done |

**Recommended:** land after 3.8a so the `DownsampleConfig` field change is in one commit.
New optional dep: `simplification>=0.7` (Rust-backed polyline simplification).

##### 3.8c — Downsampling Quality Metrics

| Phase | Task | Status |
|-------|------|--------|
| 1 | New `utils/downsampling_metrics.py` — `DownsampleMetrics` dataclass + `evaluate_downsampling()` with 3-tier memory measurement | ✅ Done |
| 2 | `benchmark_configs(data, time, configs) → pd.DataFrame` comparison table | ✅ Done |
| 3 | Segment-aware metrics (`evaluate_downsampling_segments`) — plateau vs transition RMSE via existing `binarize_signal` | ✅ Done |
| 4 | CLI: `--benchmark-downsample` flag wired into `analysis/processing.py` | ✅ Done |
| 5 | `tests/test_downsampling_metrics.py` — 17 test cases | ✅ Done |

Memory measurement tiers: Tier 1 `tracemalloc` (stdlib, default), Tier 2 subprocess RSS (`resource`, stdlib, Unix), Tier 3 `memray` (optional, Linux/macOS, captures Rust/C heap).
New optional dep group: `benchmark = ["memray>=1.0", "psutil>=5.9", "scipy>=1.9"]`.
Can be written before 3.8a/b (works with existing `stride`/`minmax`); fully meaningful after them.

#### 3.7 Pattern Entries in `*-defs.json` 🔴 OPEN

**Goal:** Allow one JSON entry to cover hundreds of similarly-named columns (`U_0`…`U_239`)
via a `"match"` regex key. Scoped to `feelpp-defs.json`; pupitre/pigbrother defs unchanged.

**See:** Phase H of [cross-domain-comparison.prompt.md](cross-domain-comparison.prompt.md)

| Step | Task | Status | Effort |
|------|------|--------|--------|
| H1 | Two-pass `load_units_from_json()` in `magnetdata_base.py` | 🔴 Open | S |
| H2 | New `python_magnetrun/feelpp-defs.json` with `U_\d+`, `T_\d+` patterns | 🔴 Open | S |
| H3 | `FeelppMagnetData.fromfeelpp()` + `SimulationRun.from_feelpp()` default to `feelpp-defs.json` | 🔴 Open | S |
| H4 | Optional `--match` flag on `field add` CLI subcommand | 🔴 Open | S |

**Independent of all other phases** — can be done any time.

#### 3.9 Hybrid Subpackage Code Quality ⬜ OPEN

**Status:** 16 findings from cross-module review. Items B0.5/B2.5 ✅ done (Phase 2B complete); items 12/13 in Quick Wins.
**See:** [docs/hybrid_refactoring_notes.md](../docs/hybrid_refactoring_notes.md)

| Priority | Item | Effort | Risk | Status |
|---|---|---|---|---|
| S1 | Hoist `safe_float` to module level in `hybrid/kHz/fepc_reader.py` (defined at lines 298 + 435) | S | Low | 🔴 Open |
| S2 | Consolidate `_resolve_backend` into `plotting/_utils.py`; import in `hybrid/plotting.py` + `plotting/timeseries.py` | S | Low | 🔴 Open |
| M1 | Unify `log_exception` / `format_exception_location` on `log_utils.py` signature; update 6 call sites in `hybrid/cli.py`; delete copy in `hybrid/utils.py` | M | Medium | 🔴 Open |
| M2 | Standardise `range` schema to dict `{"start": …, "end": …}` in `analysis.synchronization` (`compute_lag` uses tuple, `lag_correlation` uses dict) | M | Medium | 🔴 Open |
| M3 | Deprecate `processing/correlations.py` lag functions via shims forwarding to `analysis.synchronization` | M | Low | 🔴 Open |
| M4 | Extract `_apply_cnv_calibration(data, cnv_path)` helper shared by `fepc_reader.py` and `trigger_reader.py` | M | Low | 🔴 Open |
| L1 | Extract `_plot_variable_impl` / `_plot_variables_impl` in `hybrid/plotting.py`; add `downsample` param to RMS variants | L | Medium | 🔴 Open |
| L2 | Extract `_BinaryFileReaderBase` for `RMSFileReader` / `VProcessFileReader`; merge dataclasses into `ChannelVariable` | L | Medium | 🔴 Open |

#### 3.4 `analysis/__init__.py` Namespace ✅ COMPLETE

**Status:** Metrics and plotting removed from flat namespace
- `analysis.metrics.*` and `analysis.plotting.*` are the access paths for those sub-modules
- Remaining flat exports: config, loaders, synchronization, processing, downsampling

#### 3.5 Monolith Splitting ✅ MOSTLY COMPLETE

| File | Original Size | Status | Notes |
|------|---------------|--------|-------|
| `magnetdata.py` | 1500 lines | ✅ Done | Split into `magnetdata_base.py`, `magnetdata_pandas.py`, `magnetdata_tdms.py`; facade now 235 lines |
| `python_magnetrun.py` → `cli.py` | 1300 lines | 🟡 Partial | Renamed; args extracted; body still needs splitting into `commands/` |

#### 3.6 Type Hints 🟡 ONGOING

| Task | Status | Notes |
|------|--------|-------|
| 100% type hints on new code | ✅ Enforced | `ruff` enforces on new/modified code |
| Backfill `magnetdata.py` | 🔴 Open | Public API needs type hints |
| Backfill `MagnetRun.py` | 🔴 Open | Public API needs type hints |
| Backfill `analysis/plotting.py` | 🔴 Open | Public API needs type hints |
| Enable `mypy` in CI | 🔴 Open | Commented out in `.pre-commit-config.yaml` |

---

### Stream 4: Advanced Features (Future)

#### 4.1 `HybridData` Timestamp Support 🔴 OPEN

**Goal:** Add timestamp awareness to `HybridData` class

**Requirements:**
- Add `start_timestamp`, `end_timestamp` fields
- Implement `addTime()` method
- Add `getStartDate()`, `getDuration()` methods
- Required before `HybridRun` can participate in `ComparisonSession`

**Prerequisite:** `analysis/` Phase 6 (`add_time_columns` utility) — ✅ **complete**
**See:** [hybriddata-timestamp-plan.md](hybriddata-timestamp-plan.md)
**Effort:** ~0.5 days
**Status:** Unblocked — ready to implement

#### 4.2 Cross-Domain Comparison (Phases D-G) 🟡 PARTIAL

**Phase Status:**
- ✅ Phase A0-A3: Protocol extension (commit `de9f374`)
- ✅ Phase B: `SimulationRun` adapter (commit `fd83fe5`, `simulation/simulation_run.py`)
- ✅ Phase C: `BFieldRun` adapter (commit `fd83fe5`, `bfield/bfield_run.py`)
- 🔴 Phase H: Pattern entries in `*-defs.json` + `feelpp-defs.json` (independent — see Stream 3.7)
- 🔴 Phase D: Extend `*-defs.json` with simulation/bfield aliases; `KeyMapping`
- 🔴 Phase E: `ComparisonSession` — cleaner now that Stream 3.6 R4 is done (`HybridData` in hierarchy)
- 🟡 Phase F: `magnetrun compare` CLI — ✅ stub registered (`f87e8ce`, `c13c179`); handler raises `NotImplementedError` until Phase E complete
- 🔴 Phase G: Comprehensive tests

**Dependencies:** HybridData timestamp support; Phase E significantly cleaner after Stream 3.6 R4
**See:** [cross-domain-comparison.prompt.md](cross-domain-comparison.prompt.md)
**Effort:** ~2-3 weeks

#### 4.5 TDMS Export 🔴 OPEN

**Goal:** Enable saving pupitre and hybrid data as TDMS files compatible with `TdmsMagnetData` (pigbrother format), for archival and round-trip re-reading via existing TDMS infrastructure.

**See:** [pupitre_to_tdms_export.md](pupitre_to_tdms_export.md), [hybrid_to_tdms_export.md](hybrid_to_tdms_export.md)

**Prerequisite:** `HybridData.field_meta` initialisation bug — ✅ **fixed by Stream 3.6 R4** (`HybridData` now inherits `MagnetDataBase.__init__` which sets `self.field_meta = {}`).

##### Pupitre → TDMS (`PandasMagnetData.to_tdms()`)

| Step | Task | Status |
|------|------|--------|
| 1 | Add `"_tdms_groups"` key to `pupitre-defs.json` (channel group mapping) | 🔴 Open |
| 2 | Add `load_tdms_groups()` + `get_pigbrother_channel_name()` to `field_defs.py` | 🔴 Open |
| 3a | Deduplication of repeated timestamps at end of `PandasMagnetData.addTime()` | 🔴 Open |
| 3b | New `PandasMagnetData.to_tdms(filename, defs_file, groups)` method | 🔴 Open |

Key design decisions:
- 1 Hz resampling via `df.resample("1s").mean()` — fills gaps with NaN, guarantees uniform `wf_increment = 1.0`
- Columns `t`, `timestamp`, `Date`, `Time` excluded from output
- Channel name: `aliases.pigbrother` channel part if available, else column name
- Unassigned columns → fallback `"Pupitre"` group

##### Hybrid → TDMS (`HybridData.to_rms_tdms()` + `to_khz_tdms()`)

| Step | Task | Status |
|------|------|--------|
| 1 | Add `"_tdms_groups_rms"` + `"_tdms_groups_khz"` to `hybrid-defs.json` | 🔴 Open |
| 2 | Add `load_tdms_groups_rms()`, `load_tdms_groups_khz()`, `get_hybrid_channel_name()` to `field_defs.py` | 🔴 Open |
| 3 | `HybridData.to_rms_tdms()` — concatenates all RMS files, resamples per system frequency | 🔴 Open |
| 4 | `HybridData.to_khz_tdms()` — exports kHz channels; `hours=` parameter strongly recommended (1 day = ~86 M samples/channel) | 🔴 Open |

Key design decisions:
- RMS: `wf_increment = 1/frequency` per system (from `RMSFileReader.metadata["frequency"]`)
- kHz: `wf_increment = 0.001` (fixed 1 kHz); `wf_start_time` from `compute_hour_t0()` (first bin file)
- Unassigned channels → fallback group named after FEPC system (`"FEPC-AUX-LNCMI"` etc.)
- Both methods share the `_tdms_groups_*` + `aliases.pigbrother` channel-naming logic

**Effort:** M + M + M (three independent methods)
**Dependencies:** `addTime()` stable (pupitre); `HybridData.field_meta` fix (hybrid)

#### 4.6 Trigger & VProcess Integration into `HybridData` 🔴 OPEN

**Goal:** Add `read_trigger_variable`, `read_vprocess_variable`, `plot_trigger_variable`, `plot_vprocess_variable` to the `HybridData` interface ([hybrid/hybrid_data.py](../python_magnetrun/hybrid/hybrid_data.py))

Currently trigger and vprocess readers exist and work independently but are unreachable via the unified `HybridData` API (notes item 10).

**Depends on:** Stream 3.9 L2 (`_BinaryFileReaderBase`) — avoids duplicating read logic a third time
**Effort:** XL
**See:** [docs/hybrid_refactoring_notes.md](../docs/hybrid_refactoring_notes.md) item 10

#### 4.3 Pipeline Redesign (polars/narwhals) 🔴 DEFERRED

**Goal:** Eliminate double-load, use modern dataframe libraries

**Phases:**
1. Custom npTDMS with polars backend
2. narwhals wrapping in `getData()`
3. Pipeline restructure
4. Internal migration

**See:** [mrun-cache-implementation.plan.md](mrun-cache-implementation.plan.md)
**Effort:** XL (~4-6 weeks, multi-phase)
**Priority:** Low (performance optimization; package functional without it)

#### 4.4 HoloViews Plotting Migration 🔴 OPTIONAL

**Goal:** Replace 3-backend system with HoloViews + Panel + datashader

**Trade-offs:**
- ✅ Simpler downsampling integration
- ✅ Better interactive plots
- ❌ Replaces stable existing system
- ❌ New dependency

**See:** [holoviews-migration.plan.md](holoviews-migration.plan.md)
**Effort:** ~8 days
**Status:** Optional alternative path

---

## Completed Major Work (2026 Q1-Q2, updated 2026-06-09)

### ✅ Critical Issues — RESOLVED

1. **Housing/sensor-role config consolidation**
   - `housing_config.py` single source of truth
   - `field_mappings.py` deleted
   - `prepareData_legacy` removed entirely
   - `runetl.prepareData` fully driven by `HousingConfig`

2. **`MagnetData` factory refactoring**
   - `load_magnetdata(filename, defs_file)` factory entry point
   - No shim class masquerading as subclass
   - `isinstance` checks reliable

3. **Timestamp convention**
   - Both `PandasMagnetData` and `TdmsMagnetData` store naive UTC
   - Consistent handling across subclasses

4. **Protocol consolidation & unified data interface (Phase 2A)**
   - `DataLoader` single protocol
   - Both `MagnetRun` and `HybridRun` satisfy protocol
   - Cross-domain Phase A0–A3 complete

5. **Downsampling refactoring**
   - Shared `utils/downsampling.py` module
   - `DownsampleConfig` dataclass
   - `tsdownsample` in extras

6. **Plotting refactoring**
   - `python_magnetrun/plotting/` subpackage
   - `PlottingBackend` protocol + 3 implementations
   - Label/legend uniformization
   - Field-style support (commit `e26a9dd`)

7. **File validation infrastructure**
   - `utils/validation.py` with `FileFormatError`
   - Integrated in all loaders
   - 34 test methods across 10 test classes

8. **Logging infrastructure**
   - `log_utils.py` with structured logging
   - Migration ongoing (~100-200 `print()` calls remain)

9. **Lazy loading (on-demand data loading)**
   - `PandasMagnetData`: `_ensure_data_loaded()` method; full CSV read deferred until first `Data` access
   - `TdmsMagnetData`: `_LazyGroupDict` — per-group loading deferred to `__getitem__`
   - `_validate_start_timestamp` accesses `self._data` directly (avoids early full load)

10. **`Data` promoted to abstract property on ABC**
    - `Data` getter/setter declared abstract on `MagnetDataBase`; both subclasses implement via `_data` backing attr
    - `__getattribute__` override removed from `PandasMagnetData`
    - `close()` + context-manager (`__enter__`/`__exit__`) added to base class

11. **Resilient pupitre file loading** — see `truncated-pupitre-files.plan.md`
    - Encoding fallback: UTF-8 → Latin-1 in `fromtxt` and `fromcsv`
    - `on_bad_lines="warn"` on all `pd.read_csv` calls
    - `check_pupitre_truncation()` in `utils/validation.py`
    - `FileFormatError` raised for header-only files
    - `UnicodeDecodeError` added to `loaders.py` catch blocks

12. **`addData`/`computeData` metadata parameters**
    - Both methods now accept `symbol`, `unit`, `label`, `description`
    - On success, store `FieldMeta` in `self.field_meta[key]`
    - Housing-config formula maps and JSON defs now carry all four keys
    - `commands/add.py` passes keyword metadata; `examples/bilan.py` and `examples/get-record.py` updated

13. **`HybridRun.getData` formula-key resolution** — see `hybrid-formula-key-resolution.plan.md`
    - `_resolve_hybrid_formula()` helper: parses `LHS = op1 + op2 + …`, maps to `kHz/…`, returns element-wise sum
    - Guard inserted before type/system parse block in `getData`; result is cached
    - Covers `FEPC-AUX-LNCMI/ALIM1` and `ALIM2` keys used in M8 housing config

14. **`Ih`/`Ib` defined via `Idcct` in housing configs**
    - `housing_config.py` simplified: `get_pupitre_rename_map()` now derives `Ih`/`Ib` from `Idcct`
    - Updated `M8/M9/M10-housing-config.json`

15. **`hybrid/` subpackage refactoring — all 6 phases** *(branch `rework_analysis`)*
    - Phase 1: `print()` → `logger.debug/info`; commented debug removed; `print_summary()` kept
    - Phase 2: outlier duplicate functions deleted from `hybrid/utils.py`; re-exports via `from ..outliers import …`
    - Phase 3: `OUTLIER_DEFAULTS` dict in `python_magnetrun/outliers.py`; `threshold if … is not None else …` bug fixed; `OutlierDetector.__init__` updated
    - Phase 4: `OutlierConfig` frozen dataclass; `hybrid_data.py` four plot methods use `outlier_config: OutlierConfig | None`; `create_outlier_parser`/`args_to_outlier_config` in `cli_args.py`; canonical code in `python_magnetrun/outliers.py`; `hybrid/outliers.py` is a backward-compat shim
    - Phase 5: `normalize_signal`, `binarize_signal`, `_otsu_threshold` → `python_magnetrun/processing/signal.py`; `processing/__init__.py` re-exports; `hybrid/utils.py` and `hybrid/hybrid_run.py` updated; `merge_data` added to `analysis/loaders.py` (fixes prior test failure: 833 → 866 tests)
    - Phase 6: `_evict_oldest_cache_entry()` extracted with LRU docstring; all-NaN guard in `read_khz_variable`; file-existence guard + `FileNotFoundError` in `read_rms_variable`; `load_khz_config` raises `FileNotFoundError`; `_build_groups` try/except with warning; `saveData` guards group keys; 866 tests pass

16. **Outlier deduplication** *(branch `rework_analysis`)*
    - `examples/outliers.py` deleted (213-line inline rolling-MAD, never imported)
    - `processing/hysteresis.py::remove_outliers`: ~120 lines inline IQR/zscore/MAD/isolation_forest → ~15-line delegator calling `detect_outliers()` from `hybrid/outliers.py`
    - `tests/test-anomalies.py` + `tests/test-anomalies-optimized.py` deleted (CLI scripts requiring real TDMS files)
    - `tests/test_outliers.py` created: 142 tests across `TestOutlierDetector`, `TestOutlierResult`, `TestHelpers`; all synthetic data, no file I/O
    - `OutlierMethod.ISOLATION_FOREST` added to `hybrid/outliers.py`: sklearn `IsolationForest` backend, contamination threshold (default 0.1), rolling-window correctly rejected with `ValueError`
    - `_VALID_METHODS` in `hysteresis.py` extended with `"isolation_forest"`

17. **Phase 2B: Time Alignment Layer — all tasks complete** *(commits `ca4b41a`…`fb79656`)*
    - B0.5: `utc_hour_to_local()` in `hybrid/utils.py`; `read_khz_variable`/`read_rms_variable` use UTC `hours` directly; `_utc_hour_to_local` closure removed from `analysis/processing.py`
    - B1: `_khz_first_last_utc(hdata)` in `hybrid_data.py`; `HybridRun.get_time_range()` now calls it for accurate `(t_start, t_end)`
    - B2: RMS time origin is now seconds since UTC midnight of recording date
    - B2.5: `plot_rms_variable` uses stashed `orig_data`/`orig_time` in both highlight branches
    - B3: `align_to_common_time(sources, reference, hours)` in `utils/timestamps.py`; exported via `utils/__init__.py`
    - B4: `examples/plot_hybrid_with_pupitre_tdms.py` refactored to use `align_to_common_time()`

18. **`BinarizeConfig` dataclass** *(commit `02c7783`)*
    - `method`, `tolerance`, `n_bins`, `normalize`, `noise_percentile` fields
    - `LoadOptions.binarize_config: BinarizeConfig | None` field
    - Wired into `getData` flow via `_apply_voltage_mask`

16. **`analysis/` subpackage refactoring — all 6 phases** *(branch `rework_analysis`)*
    - Phase 1: dead code, logging, `DIR_*` constants centralised in `utils/files.py`; `pigbrother.py` imports `DIR_DEFAULT`/`DIR_SPIKE`
    - Phase 2: `DownsampleConfig` adopted in `analysis/plotting.py`; old percent-based helpers removed
    - Phase 3: `utils/files.py` canonical for `_open_text_with_fallback`, `extract_data`, `find_files`, `select_files`, `load_df`, `load_data`, `merge_data`; `loaders.py` imports from it; `magnetdata_pandas.py` updated
    - Phase 4: `HousingConfig` gains `get_pupitre_current_channel`, `get_pupitre_group_keys`, `get_pupitre_flow_keys`, `get_hybrid_group_keys`; `_get_pupitre_channel/group/flow`, `_get_hybrid_channel/group` deleted from `processing.py`; stray `print()` → `logger.debug()`
    - Phase 5: `FileDiscovery.discover()` → 5 private helpers; `process_overview_file()` → 3 helpers; `analysis/cli.main()` → 6 helpers (`_setup_logging`, `_collect_input_files`, `_load_records`, `_combine_dataframes`, `_run_combined_analysis`, `_emit_metrics`)
    - Phase 6: `add_time_columns(df, t0, sampling_rate=0.0)` in `utils/timestamps.py`; `add_time_column_with_offset` and `add_time_column` delegate to it; inline lambdas in `load_incident_data`, `synchronize_data`, `apply_lag_correction` replaced; `TIME_OFFSET_INCIDENTS`/`get_time_offset` removed from `processing.py` imports

---

## Quick Wins — Immediate Value Items

| Task | Effort | Impact | Status | Action |
|------|--------|--------|--------|--------|
| Fix bare `except:` clauses | 30 min | High | ✅ Done | — |
| Add `ruff` pre-commit hook | 1 hour | High | ✅ Done | — |
| File validation | 4 hours | High | ✅ Done | — |
| Add assertions to `test_python_magnetrun.py` | 30 min | Medium | ✅ Done | 7 assertions added |
| Remove `pigbrother-defs.json~` | 5 min | Low | ✅ Done | — |
| Add `*.json~` to `.gitignore` | 2 min | Low | ✅ Done | — |
| Audit TODOs in `requests/cli.py` (rename site→housing, geometry, Parts) | 15 min | Low | 🔴 Open | Four TODO comments that need tracking or resolution |
| Enable `mypy` pre-commit hook | 1 hour | Medium | 🔴 Open | Uncomment in `.pre-commit-config.yaml` (`ruff` already runs via pre-commit; no need to add to CI) |
| Add cross-refs between `remove_outliers` variants (`outliers.py` ↔ `hysteresis.py`) | S | Readability | 🔴 Open | Notes item 12 — no code change, docstring only |
| Rename `commands/plot.py:_handle_output` → `_save_or_show_figure` | S | Readability | 🔴 Open | Notes item 13 — avoids name collision with `hybrid/plotting.py:_handle_output` |

**Recommended order:**
1. ~~Add assertions to `test_python_magnetrun.py`~~ — ✅ Done
2. Enable `mypy` (enforces type hints)
3. Quick cleanups (audit TODOs in `requests/cli.py`)

---

## Action Items by Priority

### 🔴 Critical (Do This Week)

1. **Fix multiple-file `vs_time` regression**
   - Investigate plot timing issues
   - Test with multiple input files
   - Verify fix across data sources

2. **Enable `mypy`**
   - Uncomment the `mypy` hook in `.pre-commit-config.yaml`
   - `ruff` already runs via pre-commit (`--fix`) — no CI change needed

### 🟡 High Priority (Do This Month)

3. **Verify validation on all entry points**
   - Audit all CLI entry points
   - Ensure validation called before parsing
   - Add tests for validation paths

4. ~~**Complete Phase 2B (Time Alignment)**~~ — ✅ **DONE** (B0.5–B4 all landed; 1052 tests pass)

5. **Start Phase 2C: Extend `plot_data()` for hybrid** ← NOW UNBLOCKED
   - Add `df_hybrid` and `hybrid_channels` parameters to `analysis/plotting.plot_data()`
   - Implement hybrid plotting logic using `align_to_common_time()`
   - Add tests and examples

6. **Quick wins cleanup**
   - ~~Add assertions to `test_python_magnetrun.py`~~ — ✅ Done
   - Enable `mypy` pre-commit hook
   - Audit TODOs in `requests/cli.py`

### 🟢 Medium Priority (Do This Quarter)

7. **Phase 2D: Side-by-side comparison**
   - Extend `plot_comparison()` for multi-source
   - Implement subplot grid generation
   - Add linked time axis

8. **Logging migration**
   - Continue converting `print()` → `logger.*`
   - Track progress (target: >90% complete)

### 🔵 Lower Priority (Future Quarters)

9. ~~**Outlier deduplication**~~ — ✅ Done (see Stream 3.5)
10. ~~**hybrid/ refactoring**~~ — ✅ Done (all 6 phases; see Stream 3.2)
11. ~~**Reader/container split R1–R5**~~ — ✅ Done (all phases; see Stream 3.6; 1052 tests pass)
12. ~~**CLI consolidation**~~ — ✅ Done (see Stream 3.3; `magnetrun compare` stub registered)
13. **HybridData timestamp support** (~0.5 days; now fully unblocked — analysis/ Phase 6 + 3.6 R4 both done)
14. ~~**M4 / NaN-M4 downsampling**~~ — ✅ Done (see Stream 3.8a)
14b. ~~**RDP / VW downsampling**~~ — ✅ Done (see Stream 3.8b)
14c. ~~**Downsampling quality metrics**~~ — ✅ Done (see Stream 3.8c)
15. **Pattern entries in `*-defs.json`** (~2 hours; independent — do any time; see Stream 3.7)
15b. **Hybrid code quality — Stream 3.9** (S→XL; S items any time, L items plan first; see `docs/hybrid_refactoring_notes.md`)
16. **TDMS export — pupitre** (`to_tdms()` + deduplication in `addTime()`; ~M; independent)
17. **TDMS export — hybrid RMS** (`to_rms_tdms()`; ~M; `field_meta` prereq now done)
18. **TDMS export — hybrid kHz** (`to_khz_tdms()`; ~M; `field_meta` prereq now done)
19. **Cross-domain Phases H, D, E, F, G** (~2-3 weeks; H first as it's independent)
20. **Trigger & VProcess integration into `HybridData`** (XL; Stream 4.6; depends on 3.9 L2)
21. **Type hints backfill** (ongoing)

---

## Recent Changes (Last 20 Commits)

| Commit | Task | Impact |
|--------|------|--------|
| `02c7783` | `BinarizeConfig` dataclass in `hybrid_run.py` | `method`/`tolerance`/`n_bins`/`normalize`/`noise_percentile` fields; wired into `LoadOptions.binarize_config` + `getData` flow |
| `fb79656` | Add `hours` param to `align_to_common_time()` | `utils/timestamps.py` updated; demonstrator simplified; `test-timestamp.py` +19 lines |
| `1ab50de` | Refactor demonstrator (B4) | `examples/plot_hybrid_with_pupitre_tdms.py` uses `align_to_common_time()` |
| `e67233c` | Implement B3 — `align_to_common_time()` | `utils/timestamps.py` +48 lines; `utils/__init__.py` export; `test-timestamp.py` +55 lines; `test_hybrid_api.py` +30 lines |
| `e397101` | Fix RMS time origin (B2) + `_khz_first_last_utc` (B1) | RMS time now seconds from UTC midnight; `HybridRun.get_time_range()` uses bin-file lookup; `test_hybrid_api.py` +59 lines |
| `ca4b41a` | Standardise `hours` = UTC (B0.5) | `utc_hour_to_local()` in `hybrid/utils.py`; closures removed from `analysis/processing.py`; `test_hybrid_api.py` +75 lines |
| `f87e8ce` + `c13c179` | `comparison/cli.py` stub; `main.py` wired | `magnetrun compare` now discoverable; handler raises `NotImplementedError` |
| (branch) | Reader/container split (3.6) — all 5 phases | `readers/` subpackage (8 reader classes + registry); `HybridData(MagnetDataBase)` hierarchy; `DataType.HTS = 4`; `field_meta` init bug fixed; `load_magnetdata` accepts `fmt=`; 46 new tests in `tests/readers/`; **1052 tests pass** |
| (branch) | Outlier deduplication (3.5) | `examples/outliers.py` deleted; `hysteresis.py::remove_outliers` thin-delegates; `test-anomalies*.py` deleted; `tests/test_outliers.py` (142 tests); `ISOLATION_FOREST` in `OutlierMethod`; 142 new tests pass |
| (uncommitted) | `analysis/` subpackage refactoring — all 6 phases | `utils/files.py` canonical; `HousingConfig` 4 new methods; monoliths decomposed; `add_time_columns` utility; 827 tests pass |
| f6394e2 | Change `addData`/`computeData` signature | `symbol`/`unit`/`label`/`description` → `FieldMeta`; JSON configs updated |
| 6b40ea5 | Implement data-property-abc plan | `Data` as abstract property; lazy load in contract; `close()`/context manager |
| 8c4da77 | Implement truncated-pupitre-files plan | Encoding fallback, `on_bad_lines`, truncation check, `UnicodeDecodeError` in callers |
| c6404f9 | Add dev plans and code review YAML | `data-property-abc.plan.md`, `truncated-pupitre-files.plan.md`, `hybrid-formula-key-resolution.plan.md` |
| e3786f3 | Up with tests — include hybrid data | `test-vprocess.py` (496 lines), `test-cfg-parser.py` (135 lines) |
| c390172 | Implement lazy loading of actual values | `_ensure_data_loaded` in pandas; `_LazyGroupDict` in TDMS |
| 2f78661 | Use `Idcct` to define `Ih` and `Ib` | `housing_config.py` simplified; M8/M9/M10 JSON updated |
| a0b00ed | Add info log message | Logging improvement |
| 97c8efc | Update `load_mrun` for easy data loading | Convenience method |
| 2ca2ea2 | Fix matplotlib NaN rendering bug | Plotting robustness |
| 86c45c6/76351f3 | Fix plot issues | **Regression**: `vs_time` with multiple files |
| e26a9dd | Add field-style control | Per-field plot styling |
| 5fd35f1/6d0cad5 | Add/fix plotting features | Enhancement iteration |
| ea9f27d | Add tests for show/save plot | Test coverage |
| e0afc0b | Add `load_mrun` method | Convenience loading |
| da42a6c | Update plotting labels/legends | Uniformization |
| 6d2e09b | Implement downsampling refactoring | `DownsampleConfig` |
| de9f374 | Complete Phase A0-A3 | Protocol extension |

---

## Dependencies & Blockers

**Phase 2B (Time Alignment)** ✅ complete — previously blocked:
- Phase 2C (Hybrid plotting) — **now unblocked**
- Phase 2D (Comparison view)

**analysis/ Phase 6 (Timestamp utilities)** ✅ complete — previously blocked:
- HybridData timestamp support — **now unblocked**
- Cross-domain Phases D-G

**HybridData timestamps** blocks:
- Cross-domain Phase E (`ComparisonSession`)

**Stream 3.6 R4 (`HybridData` in hierarchy)** ✅ complete — delivered:
- `HybridData` is now a proper first-class `MagnetDataBase` subclass
- `field_meta` init bug fixed for free (base class `__init__` sets it)
- TDMS export (4.5 hybrid) prerequisite resolved

**No blockers for:**
- Quick wins
- CI/CD pipeline
- Logging migration
- Type hints backfill
- HybridData timestamp support (analysis/ Phase 6 done; 3.6 R4 done)
- Stream 3.7 Phase H (pattern entries — fully independent)
- Stream 3.8a M4/NaN-M4 (no new dependency, zero risk)
- Stream 3.8b RDP/VW (new `simplification` dep; do after 3.8a)
- Stream 3.8c downsampling metrics (can start before 3.8a/b)
- CLI consolidation (3.3; analysis/cli.py decomposition already done)

---

## Success Metrics

**By End of Q2 2026:**
- [x] CI/CD pipeline running on all PRs (`test.yml` + `docs.yml`; `ruff` via pre-commit)
- [x] Phase 2B (Time Alignment) complete
- [ ] Multiple-file plotting regression fixed
- [ ] All quick wins completed

**By End of Q3 2026:**
- [ ] Phase 2B-D complete (unified multi-source plotting)
- [ ] Logging migration >90% complete
- [ ] mypy enabled in CI
- [ ] Test coverage >80%

**By End of Q4 2026:**
- [ ] HybridData timestamp support complete
- [ ] Cross-domain Phases D-G complete
- [x] analysis/ and hybrid/ refactoring complete
- [ ] 100% type hints on public APIs

---

## Related Documentation

- **[ROADMAP.md](ROADMAP.md)** — Strategic direction and 6-month timeline
- **[REVIEW.md](REVIEW.md)** — Architecture review and resolved issues
- **[CODE_REVIEW.md](CODE_REVIEW.md)** — Code quality guidelines
- **Plan files:** `*-plan.md`, `*-prompt.md` — Detailed implementation plans

---

## Notes

- **Package is production-ready** — all critical issues resolved
- **Focus areas:** Stability (CI, tests) → Unified plotting → Refactoring → Advanced features
- **Parallelization:** Logging, type hints, quick wins can run concurrently with main work streams
- **Dependencies tracked:** Phase 2B is critical path for unified plotting features
