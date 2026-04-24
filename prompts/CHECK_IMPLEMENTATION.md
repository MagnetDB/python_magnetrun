# Implementation Status — python_magnetrun

*Last updated: 2026-04-24 — branch `rework_analysis`*

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
├── field_defs.py / housing_config.py  # Config layer
├── cli.py                   # CLI entry point (renamed from python_magnetrun.py)
├── cli_args.py / args.py    # CLI argument parsing
├── commands/                # Modular CLI subcommands
├── analysis/                # Analysis pipeline
├── hybrid/                  # FEPC kHz/RMS/Trigger data
├── processing/              # Signal processing
├── plotting/                # Plotting backends & utilities
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
| CI/CD Pipeline | 🔴 Open | 2-4 hours | **High** | Add `.github/workflows/ci.yml` |
| Validation on all entry points | 🟡 Verify | 2-3 hours | **High** | Infrastructure exists; verify all paths use it |
| Logging migration | 🟡 In Progress | Ongoing | **Medium** | ~100-200 `print()` calls remain |

#### Quality & Validation ✅ Mostly Complete

| Task | Status | Commit | Notes |
|------|--------|--------|-------|
| Fix bare `except:` clauses | ✅ Done | `6253de6` | No bare excepts remain in key files |
| File format validation | ✅ Done | `3e722ec` | `utils/validation.py` integrated throughout |
| `ruff` pre-commit hook | ✅ Done | `64ea699` | Enforces consistency |
| `pathlib.Path` migration | 🟡 Partial | — | ~90 occurrences converted; ongoing |

#### Test Infrastructure ✅ Mostly Complete

| Task | Status | Details |
|------|--------|---------|
| `tests/analysis/` suite | ✅ Done | 7 test files: config, loaders, metrics, plotting, processing, sync, CLI |
| `tests/test_file_validation.py` | ✅ Done | 261 lines, 10 test classes, 34 test methods covering all validators |
| Unit tests for `magnetdata.py` | ✅ Done | Covers factory, fromtdms, fromtxt, getData, column renaming |
| Unit tests for `processing/` | ✅ Done | Pure functions: smoothers, trends, peaks, stats |
| CLI entry point smoke tests | ✅ Done | Integration tests verify clean exits |
| `test_python_magnetrun.py` assertions | 🔴 Open | Legacy test file has 0 assertions |
| CI pipeline (GitHub Actions) | 🔴 Open | Need `ruff` + `pytest` on every push |

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

#### 3.1 `analysis/` Subpackage Refactoring 🔴 OPEN

**Status:** Detailed plan exists
**See:** [analysis-subpackage-refactoring.plan.md](analysis-subpackage-refactoring.plan.md)

**Key Phases:**
1. Data loading consolidation
2. Downsampling adoption (integrate `DownsampleConfig`)
3. Function decomposition (break up monoliths)
4. Channel mapping to `HousingConfig`
5. Test coverage
6. Timestamp utilities (`add_time_columns` — prerequisite for HybridData)

**Effort:** ~5-7 days
**Priority:** Medium

#### 3.2 `hybrid/` Subpackage Refactoring 🔴 OPEN

**Status:** Detailed plan exists
**See:** [hybrid-subpackage-refactoring.plan.md](hybrid-subpackage-refactoring.plan.md)
**Prerequisite:** 3.5 Outlier Deduplication below (Phase 1 of hybrid plan becomes a thin-delegate; do that first)

**Key Phases:**
1. ~~Outlier removal deduplication~~ → handled by 3.5; only `OutlierConfig` wrapper remains here
2. `OutlierConfig` dataclass (following `DownsampleConfig` pattern)
3. `signal_processing.py` extraction
4. Test coverage

**Effort:** ~10-14 hours
**Priority:** Medium

#### 3.3 CLI Consolidation 🔴 OPEN

**Status:** Detailed plan exists
**See:** [cli-consolidation.plan.md](cli-consolidation.plan.md)

**Goal:** Reduce 8 entry points to 3:
- `magnetrun` — unified dispatcher (new `python_magnetrun/main.py`)
- `magnetrun-fetch` — renamed from `srvdata-to-magnetrun` (standalone)
- `magnetrun-config` — unchanged

**New `magnetrun` subcommands:** `info`, `add`, `plot`, `select`, `stats`, `signature` (new), `analysis`, `processing`, `hybrid`, `logparser`

**Key change:** `input_file` moves to each subcommand parser (subcommand-first argv); eliminates `_normalize_argv` hack in `cli.py`

**Coordinate:** `analysis/cli.py` pass must land together with `analysis-subpackage-refactoring.plan.md` Phase 5.3 (single branch)

**Effort:** ~1-2 days
**Priority:** Low

#### 3.5 Outlier Deduplication 🔴 OPEN *(do before 3.2)*

**Status:** Detailed plan exists
**See:** [outlier-consolidation.plan.md](outlier-consolidation.plan.md)

**Canonical module:** `hybrid/outliers.py` (complete — `OutlierDetector`, `OutlierResult`, `detect_outliers`)

**Duplicates to eliminate:**

| File | Issue | Action |
|------|-------|--------|
| `examples/outliers.py` | 213-line rolling-MAD reimplementation; never imported | Delete |
| `processing/hysteresis.py::remove_outliers` | ~120 lines inline IQR/zscore/MAD | Thin-delegate to canonical module (~25 lines) |
| `tests/test-anomalies.py` + `tests/test-anomalies-optimized.py` | CLI scripts, not pytest; require real TDMS files | Delete; replace with `tests/test_outliers.py` (synthetic data) |

**Effort:** ~4-5 hours
**Priority:** Medium (precursor to 3.2)

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

**Prerequisite:** `analysis/` Phase 6 (`add_time_columns` utility)
**See:** [hybriddata-timestamp-plan.md](hybriddata-timestamp-plan.md)
**Effort:** ~0.5 days
**Status:** Blocked by analysis/ Phase 6

#### 4.2 Cross-Domain Comparison (Phases D-G) 🟡 PARTIAL

**Phase Status:**
- ✅ Phase A0-A3: Protocol extension (commit `de9f374`)
- ✅ Phase B: `SimulationRun` adapter (commit `fd83fe5`, `simulation/simulation_run.py`)
- ✅ Phase C: `BFieldRun` adapter (commit `fd83fe5`, `bfield/bfield_run.py`)
- 🔴 Phase D: Extend `*-defs.json` with simulation/bfield aliases
- 🔴 Phase E: `ComparisonSession` implementation
- 🔴 Phase F: `magnetrun-compare` CLI
- 🔴 Phase G: Comprehensive tests

**Dependencies:** HybridData timestamp support
**See:** [cross-domain-comparison.prompt.md](cross-domain-comparison.prompt.md)
**Effort:** ~2-3 weeks

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

---

## Quick Wins — Immediate Value Items

| Task | Effort | Impact | Status | Action |
|------|--------|--------|--------|--------|
| Fix bare `except:` clauses | 30 min | High | ✅ Done | — |
| Add `ruff` pre-commit hook | 1 hour | High | ✅ Done | — |
| File validation | 4 hours | High | ✅ Done | — |
| Add assertions to `test_python_magnetrun.py` | 30 min | Medium | 🔴 Open | Add meaningful test assertions |
| Remove `pigbrother-defs.json~` | 5 min | Low | ✅ Done | — |
| Add `*.json~` to `.gitignore` | 2 min | Low | ✅ Done | — |
| Audit TODOs in `requests/cli.py` (rename site→housing, geometry, Parts) | 15 min | Low | 🔴 Open | Four TODO comments that need tracking or resolution |
| Enable `mypy` pre-commit hook | 1 hour | Medium | 🔴 Open | Uncomment in `.pre-commit-config.yaml` |
| Add CI pipeline | 2 hours | High | 🔴 Open | Create `.github/workflows/ci.yml` |

**Recommended order:**
1. Add CI pipeline (enables automated testing)
2. Add assertions to `test_python_magnetrun.py` (makes CI meaningful)
3. Enable `mypy` (enforces type hints)
4. Quick cleanups (backup file already done; audit TODOs in `requests/cli.py`)

---

## Action Items by Priority

### 🔴 Critical (Do This Week)

1. **Fix multiple-file `vs_time` regression**
   - Investigate plot timing issues
   - Test with multiple input files
   - Verify fix across data sources

2. **Add CI/CD pipeline**
   - Create `.github/workflows/ci.yml`
   - Run `ruff` + `pytest` on push
   - Set up branch protection

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
   - Add assertions to `test_python_magnetrun.py`
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

9. **analysis/ refactoring** (~5-7 days)
10. **Outlier deduplication** (~4-5 hours) — precursor to hybrid/ refactoring
11. **hybrid/ refactoring** (~10-14 hours, after item 10)
12. **CLI consolidation** (~1-2 days, coordinate analysis/cli.py with item 9 Phase 5.3)
13. **HybridData timestamp support** (~0.5 days)
13. **Cross-domain Phases D-G** (~2-3 weeks)
14. **Type hints backfill** (ongoing)

---

## Recent Changes (Last 20 Commits)

| Commit | Task | Impact |
|--------|------|--------|
| a0b00ed | Add info log message | Logging improvement |
| 97c8efc | Update `load_mrun` for easy data loading | Convenience method |
| 2ca2ea2 | Fix matplotlib NaN rendering bug | Plotting robustness |
| 86c45c6/76351f3 | Fix plot issues | **Regression**: `vs_time` with multiple files |
| e26a9dd | Add field-style control | Per-field plot styling |
| 5fd35f1/6d0cad5 | Add/fix plotting features | Enhancement iteration |
| ea9f27d | Add tests for show/save plot | Test coverage |
| e0afc0b | Add `load_mrun` method | Convenience loading |
| 033752a | Update plotting features | Enhancement |
| 6255173 | Fix logging and pupitre detection | Bug fixes |
| 9d05aa0 | Fix pylance warnings | Type checking |
| da42a6c | Update plotting labels/legends | Uniformization |
| 6d2e09b | Implement downsampling refactoring | `DownsampleConfig` |
| de9f374 | Complete Phase A0-A3 | Protocol extension |

---

## Dependencies & Blockers

**Phase 2B (Time Alignment)** blocks:
- Phase 2C (Hybrid plotting)
- Phase 2D (Comparison view)

**analysis/ Phase 6 (Timestamp utilities)** blocks:
- HybridData timestamp support
- Cross-domain Phases D-G

**HybridData timestamps** blocks:
- Cross-domain Phase E (`ComparisonSession`)

**No blockers for:**
- Quick wins
- CI/CD pipeline
- Logging migration
- Type hints backfill
- analysis/ refactoring (Phases 1-5)
- Outlier deduplication (3.5)
- hybrid/ refactoring (3.2, after outlier dedup)
- CLI consolidation (3.3, coordinate analysis/cli.py with analysis Phase 5.3)

---

## Success Metrics

**By End of Q2 2026:**
- [ ] CI/CD pipeline running on all PRs
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
