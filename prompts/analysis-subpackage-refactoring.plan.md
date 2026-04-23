# Analysis Subpackage Refactoring Plan

## Context

The `analysis` subpackage was built somewhat independently from the rest of the
package. It duplicates logic already in `utils/`, has oversized functions, mixes
`print()` and `logger`, and contains dead code. This plan addresses those issues
in priority order.

---

## Phase 1 — Quick Wins (< 2 hours, zero functional risk)

### 1.1 Remove dead code

| Item | File | Lines | Action |
|------|------|-------|--------|
| `_get_archive_channel()` | `analysis/processing.py` | ~854–856 | Delete function, inline the one call site |
| Placeholder `_extract_signatures()` | `analysis/processing.py` | ~883–898 | Delete or replace with `raise NotImplementedError` |
| `ColorConfig` dataclass | `analysis/config.py` | — | Delete (never instantiated) |
| Commented-out print blocks | `analysis/cli.py` | ~334–337 | Delete |

### 1.2 Replace `print()` with `logger`

Replace all ~18 bare `print()` calls so that `--quiet` / `--debug` flags work
correctly end-to-end.

**`analysis/processing.py`** — lines approx: 418, 426, 428, 474, 476, 479, 485,
704–706.

**`analysis/cli.py`** — lines approx: 176, 200, 202, 204, 207, 209.

Pattern:
```python
# before
print(f"Loaded {len(df_list)} archive DataFrames", flush=True)

# after
logger.debug("Loaded %d archive DataFrames", len(df_list))
```

### 1.3 Deduplicate time-offset calculation

`compute_time_offset()` in `analysis/processing.py` (~line 280) is identical to
`get_time_offset()` already in `analysis/config.py` (~line 84).

- Delete `compute_time_offset()`.
- Replace every call site with `config.get_time_offset(rate)`.

### 1.4 Centralise directory-name constants

The strings `"Fichiers_Archive"`, `"Fichiers_Default"`, `"Fichiers_Manuel_Trig"`,
`"Fichiers_Spike"` are hardcoded in both `analysis/loaders.py` and
`utils/files.py`.

- Add them as module-level constants in `analysis/config.py` (or a dedicated
  `analysis/constants.py`).
- Update both files to import and use the constants.

---

## Phase 2 — Unify Downsampling API (half day)

Two parallel APIs exist:

| API | Location | Style |
|-----|----------|-------|
| `downsample_for_plot`, `downsample_dataframe`, `downsample_minmax` | `analysis/plotting.py:58–148` | percent-based |
| `DownsampleConfig` + helpers | `utils/downsampling.py` | config-based |

`analysis/processing.py` already imports `DownsampleConfig` from utils, making
the duplication visible.

### Steps

1. Extend `utils/downsampling.py` if any capability from `plotting.py` is
   missing (e.g., minmax windowing).
2. Rewrite `analysis/plotting.py` downsampling calls to use `DownsampleConfig`.
3. Delete the three percent-based functions from `plotting.py`.
4. Run existing tests; fix any regressions.

---

## Phase 3 — Consolidate Data Loading (1–2 days)

`analysis/loaders.py` (1 158 lines) reimplements logic that already lives in
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

Four helper functions in `analysis/processing.py` map logical channel keys to
physical channel names using `HousingConfig`:

- `_get_pupitre_channel(cfg, key)`
- `_get_pupitre_group(cfg, key)`
- `_get_pupitre_flow(cfg, key)`
- `_get_archive_channel(key)` — already deleted in Phase 1

### Steps

1. Add `get_pupitre_channel(key)`, `get_pupitre_group(key)`,
   `get_pupitre_flow(key)` as methods on `HousingConfig` (in
   `python_magnetrun/housing_config.py` or wherever the class lives).
2. Update all call sites in `processing.py` to use `cfg.get_pupitre_channel(key)`.
3. Delete the module-level helper functions.

---

## Phase 5 — Break Down Oversized Functions (1–2 days)

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

Extract into:

| New function | Responsibility |
|---|---|
| `_load_all_sources(record, cfg)` | Load archive, pupitre, incidents DataFrames |
| `_synchronize_sources(record, cfg)` | Apply time-alignment to each source |
| `_extract_analysis(record, cfg)` | Compute metrics, signatures, regimes |

### 5.3 `main()` in `analysis/cli.py` (~315 lines)

Extract into:

| New function | Responsibility |
|---|---|
| `_resolve_files(args)` | Expand globs, filter by time range |
| `_run_processing(files, cfg)` | Loop over files, call process_experiment |
| `_emit_plots(results, args)` | Show or save all requested plots |
| `_emit_metrics(results, args)` | Print or export distance / DTW results |

---

## Phase 6 — Standardise Time Column Creation (half day)

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

| Phase | Effort | Risk |
|-------|--------|------|
| 1 — Quick wins | < 2 h | Very low |
| 2 — Downsampling | 4 h | Low |
| 3 — Data loading | 1–2 d | Medium |
| 4 — Channel mapping | 4 h | Low |
| 5 — Break down functions | 1–2 d | Medium |
| 6 — Time columns | 4 h | Low |
| **Total** | **~5–7 days** | |

Phases 1 and 2 can be done independently and committed immediately.
Phases 3–6 should be done on a feature branch with incremental commits.
