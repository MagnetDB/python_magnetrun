# Analysis Subpackage Refactoring Plan

## Context

The `analysis` subpackage was built somewhat independently from the rest of the
package. It duplicates logic already in `utils/`, has oversized functions, mixes
`print()` and `logger`, and contains dead code. This plan addresses those issues
in priority order.

---

## Phase 1 — Quick Wins (< 2 hours, zero functional risk)

### 1.1 Remove dead code

| Item | File | Lines | Status | Action |
|------|------|-------|--------|--------|
| `_get_archive_channel()` | `analysis/processing.py` | ~825 | 🔴 Open | Delete function, inline the one call site |
| Placeholder `_extract_signatures()` | `analysis/processing.py` | ~847 | 🔴 Open | Delete or replace with `raise NotImplementedError` |
| `ColorConfig` dataclass | `analysis/config.py` | ~160 | 🔴 Open | Delete (never instantiated) |
| Commented-out print blocks | `analysis/cli.py` | — | ⬜ Check | Delete if still present |

### 1.2 Replace `print()` with `logger` ✅ COMPLETE

All bare `print()` calls in `analysis/` have been migrated to structured logging.
The `--quiet` / `--debug` flags now work correctly end-to-end.

### 1.3 Deduplicate time-offset calculation 🔴 OPEN

`compute_time_offset()` in `analysis/processing.py` (line 279) is identical to
`get_time_offset()` already in `analysis/config.py` (line 84).

**Action:**
- Delete `compute_time_offset()`.
- Replace every call site with `config.get_time_offset(rate)`.

### 1.4 Centralise directory-name constants 🔴 OPEN

The strings `"Fichiers_Archive"`, `"Fichiers_Default"`, `"Fichiers_Manuel_Trig"`,
`"Fichiers_Spike"` are hardcoded in both `analysis/loaders.py` and
`utils/files.py`.

**Action:**
- Add them as module-level constants in `analysis/config.py` (or a dedicated
  `analysis/constants.py`).
- Update both files to import and use the constants.

---

## Phase 2 — Complete Downsampling Migration (2-3 hours)

**Status:** 🟡 PARTIAL — Infrastructure in place (commit `6d2e09b`), migration incomplete

`utils/downsampling.py` with `DownsampleConfig` was created and is now used in
some parts of the codebase. However, the old percent-based functions in
`analysis/plotting.py` are still present and actively used.

### Current State

| API | Location | Status |
|-----|----------|--------|
| `DownsampleConfig` + helpers | `utils/downsampling.py` | ✅ Exists, imported in `plotting.py:53` |
| `downsample_for_plot` | `analysis/plotting.py:62` | 🔴 Still used in tests, line 400 |
| `downsample_dataframe` | `analysis/plotting.py:120` | 🔴 Still called at line 400 |
| `downsample_minmax` | `analysis/plotting.py:152` | 🔴 Still exists |

### Remaining Steps

1. ✅ ~~Extend `utils/downsampling.py`~~ — Already has needed capabilities
2. 🔴 Rewrite remaining `downsample_*()` calls to use `DownsampleConfig`
   - Main usage at `plotting.py:400`
   - Update test files that import these functions
3. 🔴 Delete the three percent-based functions from `plotting.py`
4. 🔴 Update tests in `tests/analysis/test_plotting.py`
5. 🔴 Run existing tests; fix any regressions

---

## Phase 3 — Consolidate Data Loading (1–2 days)

**Status:** 🔴 OPEN

`analysis/loaders.py` (1228 lines, grown from 1158) reimplements logic that already lives in
`utils/files.py`.

| Duplicated method | loaders.py approx line | utils/files.py approx line | Overlap |
|-------------------|------------------------|---------------------------|---------|
| `load_df` | 628 | present | 70% |
| `load_data` | 730 | present | 80% |
| `merge_data` | 775 | present | 100% |
| `find_files` | 470 | 192 | 60% |
| `select_files` | 551 | 238 | 70% |
| `extract_data` | 329 | 127 | 65% |

Also: `convert_to_timestamp()` at `loaders.py:63` reimplements timestamp parsing
already in `utils/timestamps.py` (`parse_filename_timestamp()`).

### Steps

1. Audit the signatures of each pair. Note any analysis-specific parameter that
   must be preserved.
2. Extend the `utils/` versions where needed (keep them generic).
3. Replace the bodies of the `loaders.py` functions with thin wrappers that
   delegate to `utils/`.
4. Replace `convert_to_timestamp()` with `utils.timestamps.parse_filename_timestamp()`.
5. When wrappers become one-liners, delete them and update call sites to import
   from `utils/` directly.
6. Run the full test suite.

---

## Phase 4 — Move Channel Mapping to `HousingConfig` (half day)

**Status:** 🔴 OPEN

Four helper functions in `analysis/processing.py` map logical channel keys to
physical channel names using `HousingConfig`:

- `_get_pupitre_channel(cfg, key)`
- `_get_pupitre_group(cfg, key)`
- `_get_pupitre_flow(cfg, key)`
- `_get_archive_channel(key)` — to be deleted in Phase 1

### Steps

1. Add `get_pupitre_channel(key)`, `get_pupitre_group(key)`,
   `get_pupitre_flow(key)` as methods on `HousingConfig` (in
   `python_magnetrun/housing_config.py` or wherever the class lives).
2. Update all call sites in `processing.py` to use `cfg.get_pupitre_channel(key)`.
3. Delete the module-level helper functions.

---

## Phase 5 — Break Down Oversized Functions (1–2 days)

**Status:** 🔴 OPEN

**Note:** Line numbers may have shifted; file is now 1228 lines (was 1158).

### 5.1 `discover()` in `analysis/loaders.py` (~225 lines, 5+ nesting levels)

Extract into:

| New function | Responsibility |
|---|---|
| `_parse_overview_filename(path)` | Derive run id and timestamps from filename |
| `_extract_time_range(overview_df)` | Compute start/end from loaded data |
| `_find_archive_files(root, time_range)` | Glob archive directories |
| `_find_incidents(root, time_range)` | Glob incident files |
| `_discover_hybrid_data(root, time_range)` | Handle hybrid/kHz data |

`discover()` becomes the orchestrator that calls these five.

### 5.2 `process_overview_file()` in `analysis/processing.py` (~169 lines)

**Note:** File is now 1049 lines; specific line numbers may have shifted.

Extract into:

| New function | Responsibility |
|---|---|
| `_load_all_sources(record, cfg)` | Load archive, pupitre, incidents DataFrames |
| `_synchronize_sources(record, cfg)` | Apply time-alignment to each source |
| `_extract_analysis(record, cfg)` | Compute metrics, signatures, regimes |

### 5.3 `main()` in `analysis/cli.py` (~315 lines)

**Note:** File is now 404 lines total; specific line numbers may have shifted.

Extract into:

| New function | Responsibility |
|---|---|
| `_resolve_files(args)` | Expand globs, filter by time range |
| `_run_processing(files, cfg)` | Loop over files, call process_experiment |
| `_emit_plots(results, args)` | Show or save all requested plots |
| `_emit_metrics(results, args)` | Print or export distance / DTW results |

---

## Phase 6 — Standardise Time Column Creation (half day)

**Status:** 🔴 OPEN

`"timestamp"` (Timestamp) and `"t"` (float seconds) columns are added inline in
at least three places: `processing.py`, `synchronization.py`, `loaders.py`.

### Steps

1. Add `add_time_columns(df, t0, sampling_rate) -> DataFrame` to
   `utils/timestamps.py`.
2. Replace all inline equivalents with calls to this utility.

---

## Validation Checklist (after each phase)

- [ ] `pytest tests/` passes with no regressions
- [ ] `pytest tests/analysis/` specifically passes
- [ ] `python -m python_magnetrun.analysis --help` works
- [ ] A representative run with `--show` produces correct plots
- [ ] `--quiet` suppresses all output (no stray prints)
- [ ] `--debug` shows structured log lines (not prints)

---

## Estimated Effort

| Phase | Effort | Risk | Status |
|-------|--------|------|--------|
| 1 — Quick wins | < 2 h | Very low | 🟡 Partial (1.2 done, 1.1/1.3/1.4 open) |
| 2 — Downsampling | 2-3 h | Low | 🟡 Partial (infrastructure done, migration incomplete) |
| 3 — Data loading | 1–2 d | Medium | 🔴 Open |
| 4 — Channel mapping | 4 h | Low | 🔴 Open |
| 5 — Break down functions | 1–2 d | Medium | 🔴 Open |
| 6 — Time columns | 4 h | Low | 🔴 Open |
| **Total** | **~5–7 days** | | **~1h done, ~4.5–6.5 days remaining** |

**Progress Notes:**
- Phase 1.2 (logging migration) completed across the codebase
- Phase 2 infrastructure (DownsampleConfig) in place but old functions still used
- File sizes have grown; line number references updated where possible

**Recommended Next Steps:**
1. Complete Phase 1 (< 2 hours): Remove dead code, deduplicate time-offset
2. Complete Phase 2 (2-3 hours): Finish downsampling migration
3. Phases 3–6 should be done on a feature branch with incremental commits
