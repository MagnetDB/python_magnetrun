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

### 1.3 Deduplicate time-offset calculation 🔴 OPEN

`compute_time_offset()` in `analysis/processing.py` (line 290) is identical to
`get_time_offset()` already in `analysis/config.py` (line 84).

**Action:**
- Delete `compute_time_offset()`.
- Replace every call site with `config.get_time_offset(rate)`.

### 1.4 Centralise directory-name constants 🔴 OPEN

The strings `"Fichiers_Archive"`, `"Fichiers_Default"`, `"Fichiers_Manuel_Trig"`,
`"Fichiers_Spike"` are hardcoded in both `analysis/loaders.py` (lines 668–674)
and `utils/files.py` (lines 242–247). `utils/files.py` also uses a dict mapping
(`"Archive"` → `"Fichiers_Archives"`, etc.) at lines 45–48.

**Action:**
- Add them as module-level constants in `analysis/config.py` (or a dedicated
  `analysis/constants.py`).
- Update both files to import and use the constants.

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

**Status:** 🔴 OPEN

`analysis/loaders.py` (1403 lines) reimplements logic that already lives in
`utils/files.py`.

| Duplicated method | loaders.py approx line | utils/files.py approx line | Overlap |
|-------------------|------------------------|---------------------------|---------|
| `load_df` | 786 | present | 70% |
| `load_data` | 876 | present | 80% |
| `merge_data` | 921 | present | 100% |
| `find_files` | 616 | 192 | 60% |
| `select_files` | 697 | 238 | 70% |
| `extract_data` | 462 | 127 | 65% |

Also: `convert_to_timestamp()` at `loaders.py:64` reimplements timestamp parsing
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

**Status:** 🟡 PARTIAL — `HousingConfig` has related lookup methods; processing.py helpers not yet removed

`HousingConfig` (housing_config.py:232–260) now has `get_pupitre_channel()`,
`get_hybrid_channel()`, `get_flow_channel()`, `get_rpm_channel()`, `get_pin_channel()`.
However, three module-level helpers in `analysis/processing.py` that wrap similar
lookups have not been removed or updated to delegate:

- `_get_pupitre_channel(cfg, key)` — processing.py:997
- `_get_pupitre_group(cfg, key)` — processing.py:1006
- `_get_pupitre_flow(cfg, key)` — processing.py:1024

### Steps

1. ✅ ~~Add `get_pupitre_channel(key)` to `HousingConfig`~~ — Done (housing_config.py:232)
2. 🔴 Verify `_get_pupitre_channel` in processing.py delegates to `cfg.get_pupitre_channel(key)`
   and update all call sites to use the `HousingConfig` method directly; delete the wrapper.
3. 🔴 Add `get_pupitre_group(key)` and `get_pupitre_flow()` to `HousingConfig` if semantics
   differ from existing methods; otherwise map call sites to the closest existing method.
4. 🔴 Delete `_get_pupitre_group` and `_get_pupitre_flow` from processing.py.
5. 🔴 Also delete `_get_archive_channel` here (or coordinate with Phase 1.1).

---

## Phase 5 — Break Down Oversized Functions (1–2 days)

**Status:** 🟡 PARTIAL (5.3 partially done; 5.1 and 5.2 open)

**Note:** File sizes as of last audit: `loaders.py` 1403 lines, `processing.py` 1293 lines, `cli.py` 589 lines.

### 5.1 `discover()` in `analysis/loaders.py` (class method, line ~1100, ~225 lines, 5+ nesting levels)

🔴 OPEN — no helpers extracted yet.

Extract into:

| New function | Responsibility |
|---|---|
| `_parse_overview_filename(path)` | Derive run id and timestamps from filename |
| `_extract_time_range(overview_df)` | Compute start/end from loaded data |
| `_find_archive_files(root, time_range)` | Glob archive directories |
| `_find_incidents(root, time_range)` | Glob incident files |
| `_discover_hybrid_data(root, time_range)` | Handle hybrid/kHz data |

`discover()` becomes the orchestrator that calls these five.

### 5.2 `process_overview_file()` in `analysis/processing.py` (line 668, ~169 lines)

🔴 OPEN — no helpers extracted yet.

Extract into:

| New function | Responsibility |
|---|---|
| `_load_all_sources(record, cfg)` | Load archive, pupitre, incidents DataFrames |
| `_synchronize_sources(record, cfg)` | Apply time-alignment to each source |
| `_extract_analysis(record, cfg)` | Compute metrics, signatures, regimes |

### 5.3 `main()` in `analysis/cli.py` (line 501)

🟡 PARTIAL — several helpers extracted; `main()` is still ~100 lines but significantly reduced.

**Already extracted (under different names than originally planned):**

| Function | Line | Responsibility |
|---|---|---|
| `_setup_logging(parsed_args)` | 83 | Configure logging, return logger |
| `_collect_input_files(parsed_args, config)` | 109 | Expand globs, return sorted file list |
| `_load_records(input_files, config, parsed_args, housing, logger)` | 120 | Loop over files, call `process_experiment`, accumulate results |
| `_combine_dataframes(all_dfs)` | 257 | Merge per-file DataFrames |
| `_run_combined_analysis(results, …)` | 296 | Metrics, distance, DTW, combined plots |

**Still in `main()` / remaining to extract:**
- Plot emission and save logic (`_emit_plots`)
- Summary / metrics export (`_emit_metrics`)

**Coordinate with `cli-consolidation.plan.md`:** adding `register(subparsers)` to
`analysis/cli.py` (per the CLI consolidation plan) and completing this decomposition
should be done in a single branch.

---

## Phase 6 — Standardise Time Column Creation (half day)

**Status:** 🔴 OPEN

`"timestamp"` (Timestamp) and `"t"` (float seconds) columns are added in two
existing module functions:

- `add_time_column_with_offset(df, t0, sampling_rate)` — `analysis/processing.py:310`
- `add_time_column(df, …)` — `analysis/synchronization.py:733`

These are subtly different and not yet unified under a shared utility.

### Steps

1. Add `add_time_columns(df, t0, sampling_rate) -> DataFrame` to
   `utils/timestamps.py`.
2. Replace both inline equivalents with calls to this utility.

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
| 1 — Quick wins | < 2 h | Very low | 🟡 Partial (1.2 done; 1.1 revised, 1.3/1.4 open) |
| 2 — Downsampling | 2-3 h | Low | ✅ Complete |
| 3 — Data loading | 1–2 d | Medium | 🔴 Open |
| 4 — Channel mapping | 4 h | Low | 🟡 Partial (`get_pupitre_channel` on `HousingConfig`; wrappers not deleted) |
| 5 — Break down functions | 1–2 d | Medium | 🟡 Partial (5.3 partially done; 5.1/5.2 open) |
| 6 — Time columns | 4 h | Low | 🔴 Open |
| **Total** | **~5–7 days** | | **~1.5 h done, ~4–6 days remaining** |

**Progress Notes:**
- Phase 1.2 (logging migration) completed across the codebase
- Phase 1.1: `ColorConfig` was NOT dead code — it is actively used as `default_factory` in `AnalysisConfig`; original note corrected
- Phase 2 infrastructure (`DownsampleConfig`) in place and used in `plot_data`/`plot_comparison`; old percent-based functions still defined
- Phase 4: `HousingConfig` already has `get_pupitre_channel` and related lookup methods; processing.py wrappers not yet removed
- Phase 5.3: partial — `_setup_logging`, `_collect_input_files`, `_load_records`, `_combine_dataframes`, `_run_combined_analysis` extracted; plot/metrics emission still in `main()`

**Recommended Next Steps:**
1. Complete Phase 1 (< 2 hours): inline `_get_archive_channel`, stub `_extract_signatures`, deduplicate time-offset (1.3), centralise directory constants (1.4)
2. Complete Phase 2 (2-3 hours): remove the three old `downsample_*` functions from `plotting.py`
3. Complete Phase 4 (4 hours): remove processing.py channel-mapping wrappers; delegate to `HousingConfig`
4. Phases 3, 5, 6 on a feature branch with incremental commits
