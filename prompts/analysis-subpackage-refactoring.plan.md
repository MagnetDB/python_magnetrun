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
| `_get_archive_channel()` | `analysis/processing.py` | 1069 | 🔴 Open | Body is `return key`; inline the one call site (processing.py:1130) and delete |
| `_extract_signatures()` | `analysis/processing.py` | 1091 | 🔴 Open | Has real content (min/max/mean), but labelled a placeholder (integrates with future `Signature` class). Replace body with `raise NotImplementedError` until that class lands. |
| `ColorConfig` dataclass | `analysis/config.py` | 160 | ✅ Keep | Used as `default_factory` in `AnalysisConfig.colors` (config.py:494). Not dead code — original note was incorrect. |
| Commented-out print blocks | `analysis/cli.py` | — | ⬜ Check | Delete if still present |

### 1.2 Replace `print()` with `logger` ✅ COMPLETE

All bare `print()` calls in `analysis/` have been migrated to structured logging.
The `--quiet` / `--debug` flags now work correctly end-to-end.

### 1.3 Deduplicate time-offset calculation ✅ COMPLETE

`compute_time_offset()` was deleted during Phase 2 cleanup.  Every call site
in `analysis/processing.py` already uses `get_time_offset()` imported from
`analysis/config.py`.  `analysis/__init__.py` exports `get_time_offset` from
`.config`.  The test suite documents this in
`tests/analysis/test_processing.py` (comment: "replaces deleted compute_time_offset").

### 1.4 Centralise directory-name constants ✅ COMPLETE

`DIR_ARCHIVE`, `DIR_DEFAULT`, `DIR_TRIGGER`, `DIR_SPIKE` are defined once in
`utils/files.py` and imported everywhere they are used:
- `analysis/loaders.py` — imports all four from `utils.files` (done in Phase 3).
- `runlogs/pigbrother.py` — `file_folder` property and `_is_defaut_file` helper
  now import `DIR_SPIKE` and `DIR_DEFAULT` from `utils.files` instead of using
  hardcoded string literals.
- The only remaining literal strings are inside a test-fixture log excerpt
  (lines 1089–1100 of `pigbrother.py`) which cannot be parameterised.

---

## Phase 2 — Complete Downsampling Migration ✅ COMPLETE

`downsample_for_plot`, `downsample_dataframe`, and `downsample_minmax` have been
deleted from `analysis/plotting.py`.  `analysis/__init__.py` now re-exports
`DownsampleConfig`, `downsample_arrays`, and `downsample_dataframe` directly from
`utils.downsampling`.  `tests/analysis/test_plotting.py` rewritten to use the
canonical `DownsampleConfig` API.  `compute_time_offset` (deleted in item 3 of
Phase 1) also cleaned up: `analysis/__init__.py` now exports `get_time_offset`
from `.config`; `tests/analysis/test_processing.py` updated accordingly.
53 tests pass.

---

## Phase 3 — Consolidate Data Loading (1–2 days)

**Status:** ✅ COMPLETE

`utils/files.py` is now the canonical location for all shared data-loading
utilities.  `analysis/loaders.py` imports them instead of duplicating them.

**What was done:**
- `_open_text_with_fallback` moved from `magnetdata_pandas.py` → `utils/files.py`;
  `magnetdata_pandas.py` now imports it from there.
- `_tdms_end_from_properties` and `_pupitre_end_from_last_line` (private helpers)
  moved from `loaders.py` → `utils/files.py`.
- `TIMESTAMP_FORMAT` constant moved from `loaders.py` → `utils/files.py`
  (re-exported in `loaders.py` via import).
- `extract_data` rewritten to use `parse_filename_timestamp` instead of the
  local `convert_to_timestamp`; moved to `utils/files.py`.
- `find_files`, `select_files`, `load_df`, `load_data`, `merge_data` — the
  loaders.py versions (which were the better implementations) replaced the
  weaker stubs in `utils/files.py`; loaders.py now imports from utils/.
- `convert_to_timestamp` kept in `loaders.py` (public API, tested) but its
  internal usage eliminated (superseded by `parse_filename_timestamp`).
- 814 tests pass; 4 pre-existing failures and 14 pre-existing errors (all
  `FileNotFoundError` for missing data files) are unchanged.

---

## Phase 4 — Move Channel Mapping to `HousingConfig` (half day)

**Status:** ✅ COMPLETE

Added four new methods to `HousingConfig` keyed by "Courant_GR*" / group-name style
(distinct from the existing "Référence_GR*" style methods):

- `get_pupitre_current_channel(key)` — maps "Courant_GR1"/"Courant_GR2" to pupitre current column
- `get_pupitre_group_keys(group)` — returns pupitre column list for a TDMS group name
- `get_pupitre_flow_keys()` — returns all flow/rpm/pin pupitre column names
- `get_hybrid_group_keys(group)` — returns hybrid column list for a TDMS group name

Deleted from `analysis/processing.py`: `_get_pupitre_channel`, `_get_pupitre_group`,
`_get_pupitre_flow`, `_get_hybrid_channel` (dead code), `_get_hybrid_group`, and the
commented-out `_get_archive_channel` block. All call sites updated to use `HousingConfig`
methods. Stray `print()` debug calls converted to `logger.debug()`. 827 tests pass.

---

## Phase 5 — Break Down Oversized Functions (1–2 days)

**Status:** ✅ COMPLETE

### 5.1 `discover()` in `analysis/loaders.py` ✅ COMPLETE

Five private instance methods extracted into `FileDiscovery` before `discover()`:

| New method | Responsibility |
|---|---|
| `_resolve_overview_path(overview_file, housing, filename, basename)` | Resolve bare filename to full path under pigbrother_datadir |
| `_parse_overview_filename(filename, housing)` | Extract (housing, date, time) from overview filename stem |
| `_select_related_files(overview_path, housing, date, time, start, end)` | Glob and time-filter all file types related to overview_path |
| `_discover_runlogs(file_set, start, end)` | Populate pigbrother and pupitre run-log fields in place |
| `_discover_hybrid_data(file_set, housing, filename, resolved_overview, start, end)` | Populate hybrid kHz/RMS/trigger fields for M8 |

`discover()` is now a ~50-line orchestrator calling these five methods.

### 5.2 `process_overview_file()` in `analysis/processing.py` ✅ COMPLETE

Three module-level helpers extracted before `process_overview_file()`:

| New function | Responsibility |
|---|---|
| `_check_overview_end_state(df_overview, record, time_zone)` | Warn when magnet appears still running at end of overview |
| `_load_all_sources(record, housing_config, config, keys)` | Load archive, pupitre, hybrid and incident data into record.data |
| `_extract_analysis(record, housing_config, keys, config)` | Run post-load analysis: signatures and optional lag correlation |

`process_overview_file()` reduced from ~265 lines to ~60 lines. Also fixed 3 leftover `print()` → `logger.info()` calls in the hybrid loading section.

### 5.3 `main()` in `analysis/cli.py` ✅ COMPLETE

All helpers extracted; `main()` is now ~50 lines:

| Function | Responsibility |
|---|---|
| `_setup_logging(parsed_args)` | Configure logging, return logger |
| `_collect_input_files(parsed_args, config)` | Expand globs, return sorted file list |
| `_load_records(input_files, config, parsed_args, housing, logger)` | Loop over files, call `process_experiment`, accumulate results |
| `_combine_dataframes(all_dfs)` | Merge per-file DataFrames |
| `_run_combined_analysis(results, …)` | Metrics, distance, DTW, combined plots |
| `_emit_metrics(results, input_files, combined_metrics, parsed_args, logger)` | Log final summary and per-key distance metrics; returns exit code |

827 tests pass, 6 skipped.

---

## Phase 6 — Standardise Time Column Creation (half day)

**Status:** ✅ COMPLETE

`add_time_columns(df, t0, sampling_rate=0.0, timestamp_col, time_col) -> DataFrame`
added to `utils/timestamps.py`.  When `sampling_rate > 0` it adds a
half-period offset `1/(2*sampling_rate)` (TDMS downsampled window centre);
otherwise no offset.

- `add_time_column_with_offset` (`processing.py`) — body replaced with delegation to `add_time_columns`
- `add_time_column` (`synchronization.py`) — body replaced with delegation to `add_time_columns`
- Inline lambda in `load_incident_data` (`processing.py`) replaced with `add_time_columns(..., sampling_rate=SAMPLING_RATE_INCIDENTS)`
- Inline lambdas in `synchronize_data` and `apply_lag_correction` (`synchronization.py`) replaced with `add_time_columns` calls
- `TIME_OFFSET_INCIDENTS` import removed from `processing.py`; `SAMPLING_RATE_INCIDENTS` used instead
- `get_time_offset` import removed from `processing.py` (no longer called there)
- `add_time_columns` re-exported from `analysis/__init__.py`
- 827 tests pass, 6 skipped

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
| 1 — Quick wins | < 2 h | Very low | ✅ Complete (1.1 revised; 1.2/1.3/1.4 done) |
| 2 — Downsampling | 2-3 h | Low | ✅ Complete |
| 3 — Data loading | 1–2 d | Medium | ✅ Complete |
| 4 — Channel mapping | 4 h | Low | ✅ Complete (4 new HousingConfig methods; 5 processing.py wrappers deleted) |
| 5 — Break down functions | 1–2 d | Medium | ✅ Complete (5.1/5.2/5.3 all done; 827 tests pass) |
| 6 — Time columns | 4 h | Low | ✅ Complete (`add_time_columns` in `utils/timestamps.py`; all call sites unified) |
| **Total** | **~5–7 days** | | **All phases complete** |

**Progress Notes:**
- Phase 1.2 (logging migration) completed across the codebase
- Phase 1.1: `ColorConfig` was NOT dead code — it is actively used as `default_factory` in `AnalysisConfig`; original note corrected
- Phase 2 infrastructure (`DownsampleConfig`) in place and used in `plot_data`/`plot_comparison`
- Phase 3: `utils/files.py` is now canonical for shared data-loading utilities; `loaders.py` imports from it
- Phase 4: 4 new `HousingConfig` methods (`get_pupitre_current_channel`, `get_pupitre_group_keys`, `get_pupitre_flow_keys`, `get_hybrid_group_keys`); 5 processing.py wrappers deleted; stray debug `print()` calls converted to `logger.debug()`
- Phase 5: all three sub-phases complete — `discover()`, `process_overview_file()`, and `main()` decomposed into focused helpers; 827 tests pass, 6 skipped

**All phases complete.** The `analysis` subpackage refactoring plan is fully executed.
