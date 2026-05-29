# Implementation Status — python_magnetrun

*Last updated: 2026-05-12 — branch `rework_analysis`*

This document tracks detailed implementation status and task completion. For strategic direction, see [ROADMAP.md](ROADMAP.md). For architectural review, see [REVIEW.md](REVIEW.md).

---

## Package Status: Production-Ready ✅

The package is stable and functional for core use cases. All critical structural issues have been resolved. Focus is now on stability improvements, unified plotting features, and internal refactoring.

---

## Package Structure

```
python_magnetrun/
├── magnetdata_base.py       # ABC
├── magnetdata_pandas.py     # Pandas impl
├── magnetdata_tdms.py       # TDMS impl
├── magnetdata.py            # Factory + backward-compat shim
├── MagnetRun.py             # Session container
├── runetl.py                # ETL helpers
├── outliers.py              # Canonical outlier detection (OutlierConfig, OutlierDetector, OUTLIER_DEFAULTS)
├── field_defs.py / housing_config.py  # Config layer
├── feelpp-defs.json         # (planned Phase H2) pattern-based defs for feelpp/paraview
├── cli.py                   # CLI entry point (renamed from python_magnetrun.py)
├── cli_args.py / args.py    # CLI argument parsing (create_outlier_parser, args_to_outlier_config)
├── commands/                # Modular CLI subcommands
├── analysis/                # Analysis pipeline
├── readers/                 # (planned Stream 3.6) pure I/O readers — one class per format
├── hybrid/                  # FEPC kHz/RMS/Trigger data (outliers.py is a backward-compat shim)
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
└── TdmsMagnetData

load_magnetdata(filename)   ← standalone factory (magnetdata.py)

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

#### Phase 2B: Time Alignment Layer 🟡 PARTIAL

| Component | Status | Notes |
|-----------|--------|-------|
| TDMS `wf_start_time` exposed | ✅ Done | `get_time_range()` via protocol |
| Pupitre naive UTC timestamps | ✅ Done | `PandasMagnetData.addTime()` converts local → UTC |
| kHz/RMS UTC conversion | 🔴 Open | Currently seconds-from-day-start; needs UTC on load |
| `align_to_common_time()` utility | 🔴 Open | Multi-source time alignment helper |

**Blockers:** kHz/RMS timestamp conversion
**Effort:** ~1-2 weeks
**Next Steps:**
1. Add UTC timestamp conversion to kHz/RMS loaders
2. Implement `align_to_common_time()` in `utils/`
3. Add tests for multi-source alignment

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

**Dependencies:** Phase 2B completion
**Effort:** ~2-3 weeks
**Status:** Awaiting Phase 2B

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

#### 3.3 CLI Consolidation 🔴 OPEN

**Status:** Detailed plan exists
**See:** [cli-consolidation.plan.md](cli-consolidation.plan.md)

**Goal:** Reduce 8 entry points to 3:
- `magnetrun` — unified dispatcher (new `python_magnetrun/main.py`)
- `magnetrun-fetch` — renamed from `srvdata-to-magnetrun` (standalone)
- `magnetrun-config` — unchanged

**New `magnetrun` subcommands:** `info`, `add`, `plot`, `select`, `stats`, `signature` (new), `analysis`, `processing`, `hybrid`, `logparser`

**Key change:** `input_file` moves to each subcommand parser (subcommand-first argv); eliminates `_normalize_argv` hack in `cli.py`

**Note:** `analysis/cli.py` function decomposition (Phase 5.3) is done — only `register(subparsers)` wiring into the unified dispatcher remains

**Effort:** ~1-2 days
**Priority:** Low

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

#### 3.6 Reader/Container Split 🔴 OPEN

**Goal:** Extract format-parsing logic from container classes into a `readers/` subpackage.
Public API unchanged; migration is incremental.

**See:** [reader-container-refactoring.plan.md](reader-container-refactoring.plan.md)

| Phase | Task | Status | Effort |
|-------|------|--------|--------|
| R1 | CSV readers extracted from `magnetdata_pandas.py` factory methods | 🔴 Open | S |
| R2 | `TdmsReader` extracted from `TdmsMagnetData._fromtdms()` | 🔴 Open | S |
| R3 | `HtsReader` + `DataType.HTS = 4` (new format: `;` sep, units-in-header) | 🔴 Open | S |
| R4 | `HybridReader` (composite) + `HybridData` joins `MagnetDataBase` hierarchy | 🔴 Open | M |
| R5 | Reader registry + `load_magnetdata()` uses registry | 🔴 Open | S |

**Key payoff of R4:** removes 4 `NotImplementedError` stubs in `HybridData`; removes all
`isinstance(data, HybridData)` branches in `processing.py` and plotting code; unblocks Phase E
by making `HybridData` a proper first-class `MagnetDataBase` subclass.

#### 3.8 Downsampling Extensions 🔴 OPEN

Three independent additions to `utils/downsampling.py`. All phases are **S** effort. No changes needed in callers (`magnetdata_pandas.py`, `magnetdata_tdms.py`, `analysis/processing.py`, `hybrid/hybrid_run.py`).

**See:** [m4-downsampling.plan.md](m4-downsampling.plan.md), [rdp-downsampling.plan.md](rdp-downsampling.plan.md), [downsampling-metrics.plan.md](downsampling-metrics.plan.md)

##### 3.8a — M4 / NaN-M4

| Phase | Task | Status |
|-------|------|--------|
| 1 | Add `m4` branch to `_downsample_indices`; import `M4Downsampler` from `tsdownsample` | 🔴 Open |
| 2 | Add `nan_m4` early-exit in `downsample_arrays` + `downsample_dataframe` (bypasses NaN-strip path) | 🔴 Open |
| 3 | Add `'m4'`, `'nan_m4'` to `DOWNSAMPLE_METHODS` in `cli_args.py` | 🔴 Open |
| 4 | `tests/test_downsampling.py` — 8 new test cases | 🔴 Open |

No new dependency: `M4Downsampler` / `NaNM4Downsampler` already in `tsdownsample`.

##### 3.8b — RDP / Visvalingam-Whyatt

| Phase | Task | Status |
|-------|------|--------|
| 1 | Add `epsilon: float \| None = None` to `DownsampleConfig`; add `rdp`/`vw` dispatch + `from_n_out_rdp()` binary-search factory | 🔴 Open |
| 2 | Add `rdp = ["simplification>=0.7"]` to `pyproject.toml` extras | 🔴 Open |
| 3 | Add `'rdp'`, `'vw'` to `DOWNSAMPLE_METHODS`; pass `epsilon` from CLI JSON params | 🔴 Open |
| 4 | `tests/test_downsampling.py` — 9 new test cases | 🔴 Open |

**Recommended:** land after 3.8a so the `DownsampleConfig` field change is in one commit.
New optional dep: `simplification>=0.7` (Rust-backed polyline simplification).

##### 3.8c — Downsampling Quality Metrics

| Phase | Task | Status |
|-------|------|--------|
| 1 | New `utils/downsampling_metrics.py` — `DownsampleMetrics` dataclass + `evaluate_downsampling()` with 3-tier memory measurement | 🔴 Open |
| 2 | `benchmark_configs(data, time, configs) → pd.DataFrame` comparison table | 🔴 Open |
| 3 | Segment-aware metrics (`evaluate_downsampling_segments`) — plateau vs transition RMSE via existing `binarize_signal` | 🔴 Open |
| 4 | CLI: `--benchmark-downsample` flag wired into `analysis/processing.py` | 🔴 Open |
| 5 | `tests/test_downsampling_metrics.py` — 17 test cases | 🔴 Open |

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

#### 3.4 `analysis/__init__.py` Namespace 🔴 OPEN

**Issue:** 80+ names exported flat

**Goal:** Split into sub-namespaces:
- `analysis.metrics`
- `analysis.plot`
- `analysis.loaders`

**Effort:** ~1 day
**Priority:** Low

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
- 🔴 Phase D: Extend `*-defs.json` with simulation/bfield aliases; `KeyMapping` — cleaner after R4
- 🔴 Phase E: `ComparisonSession` — cleaner after Stream 3.6 R4 (`HybridData` in hierarchy)
- 🔴 Phase F: `magnetrun compare` CLI
- 🔴 Phase G: Comprehensive tests

**Dependencies:** HybridData timestamp support; Phase E significantly cleaner after Stream 3.6 R4
**See:** [cross-domain-comparison.prompt.md](cross-domain-comparison.prompt.md)
**Effort:** ~2-3 weeks

#### 4.5 TDMS Export 🔴 OPEN

**Goal:** Enable saving pupitre and hybrid data as TDMS files compatible with `TdmsMagnetData` (pigbrother format), for archival and round-trip re-reading via existing TDMS infrastructure.

**See:** [pupitre_to_tdms_export.md](pupitre_to_tdms_export.md), [hybrid_to_tdms_export.md](hybrid_to_tdms_export.md)

**Prerequisite:** `HybridData.field_meta` initialisation bug — `__init__()` is missing `self.field_meta = {}`, causing `AttributeError` after `load_units_from_json()`. Fixed for free by Stream 3.6 R4 (`HybridData` joins `MagnetDataBase`); can also be patched independently in one line.

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

## Completed Major Work (2026 Q1-Q2)

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
| Enable `mypy` pre-commit hook | 1 hour | Medium | 🔴 Open | Uncomment in `.pre-commit-config.yaml` |
| Enable `mypy` pre-commit hook | 1 hour | Medium | 🔴 Open | Uncomment in `.pre-commit-config.yaml` (`ruff` already runs via pre-commit; no need to add to CI) |

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

4. **Complete Phase 2B (Time Alignment)**
   - Add UTC conversion to kHz/RMS loaders
   - Implement `align_to_common_time()` utility
   - Add multi-source alignment tests

5. **Quick wins cleanup**
   - ~~Add assertions to `test_python_magnetrun.py`~~ — ✅ Done
   - Enable `mypy` pre-commit hook
   - Audit TODOs in `requests/cli.py`

### 🟢 Medium Priority (Do This Quarter)

6. **Phase 2C: Extend plot_data() for hybrid**
   - Add `df_hybrid` and `hybrid_channels` parameters
   - Implement hybrid plotting logic
   - Add tests and examples

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
11. **CLI consolidation** (~1-2 days; `analysis/cli.py` decomposition done)
12. **HybridData timestamp support** (~0.5 days; now unblocked)
13. **M4 / NaN-M4 downsampling** (~S each; no new dep; do any time — see Stream 3.8a)
13b. **RDP / VW downsampling** (~S; new `simplification` dep; do after 3.8a — see Stream 3.8b)
13c. **Downsampling quality metrics** (~M total; independent; more useful after 3.8a/b — see Stream 3.8c)
14. **Pattern entries in `*-defs.json`** (~2 hours; independent — do any time; see Stream 3.7)
14. **Reader/container split R1–R3** (~S each; independent — no behaviour change)
15. **Reader/container split R4** (`HybridData` hierarchy; ~M; do before Phase E; also fixes `field_meta` bug)
16. **TDMS export — pupitre** (`to_tdms()` + deduplication in `addTime()`; ~M; independent)
17. **TDMS export — hybrid RMS** (`to_rms_tdms()`; ~M; requires `field_meta` fix)
18. **TDMS export — hybrid kHz** (`to_khz_tdms()`; ~M; requires `field_meta` fix)
19. **Cross-domain Phases H, D, E, F, G** (~2-3 weeks; H first as it's independent)
20. **Type hints backfill** (ongoing)

---

## Recent Changes (Last 20 Commits)

| Commit | Task | Impact |
|--------|------|--------|
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

**Phase 2B (Time Alignment)** blocks:
- Phase 2C (Hybrid plotting)
- Phase 2D (Comparison view)

**analysis/ Phase 6 (Timestamp utilities)** ✅ complete — previously blocked:
- HybridData timestamp support — **now unblocked**
- Cross-domain Phases D-G

**HybridData timestamps** blocks:
- Cross-domain Phase E (`ComparisonSession`)

**Stream 3.6 R4 (`HybridData` in hierarchy)** significantly simplifies:
- Cross-domain Phase E — removes `isinstance(data, HybridData)` branches
- `HybridRun` delegation — inherits `addData`/`saveData`/`computeData` for free
- TDMS export (4.5 hybrid) — fixes missing `self.field_meta = {}` in `HybridData.__init__()` for free

**No blockers for:**
- Quick wins
- CI/CD pipeline
- Logging migration
- Type hints backfill
- HybridData timestamp support (analysis/ Phase 6 done)
- Stream 3.6 R1–R3 (CSV/TDMS/HTS readers — additive, no behaviour change)
- Stream 3.7 Phase H (pattern entries — fully independent)
- Stream 3.8a M4/NaN-M4 (no new dependency, zero risk)
- Stream 3.8b RDP/VW (new `simplification` dep; do after 3.8a)
- Stream 3.8c downsampling metrics (can start before 3.8a/b)
- CLI consolidation (3.3; analysis/cli.py decomposition already done)

---

## Success Metrics

**By End of Q2 2026:**
- [x] CI/CD pipeline running on all PRs (`test.yml` + `docs.yml`; `ruff` via pre-commit)
- [ ] Multiple-file plotting regression fixed
- [ ] All quick wins completed
- [ ] Phase 2B (Time Alignment) complete

**By End of Q3 2026:**
- [ ] Phase 2B-D complete (unified multi-source plotting)
- [ ] Logging migration >90% complete
- [ ] mypy enabled in CI
- [ ] Test coverage >80%

**By End of Q4 2026:**
- [ ] HybridData timestamp support complete
- [ ] Cross-domain Phases D-G complete
- [ ] analysis/ and hybrid/ refactoring complete
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
