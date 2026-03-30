# Implementation Checklist — python_magnetrun

*Last updated: 2026-03-30 — branch `separate-cooling`*

Tracks progress against [ROADMAP.md](ROADMAP.md).

---

## Priority 1 — Package Correctness & Reliability

### Phase 1A — Stop the bleeding

| Task | Status | Notes |
|------|--------|-------|
| Fix 5 bare `except:` clauses | ✅ Done | `6253de6` — no bare excepts remain in key files |
| Add `ruff` pre-commit hook | ✅ Done | `64ea699` — `.pre-commit-config.yaml` in place |
| Add file-format validation before parsing | ⬜ Pending | `analysis/loaders.py`, `magnetdata.py` |
| Replace `print()` with `logger.*` | ⬜ Pending | 1477 `print()` calls remain; 706 logger usages already present |
| Migrate file paths to `pathlib.Path` | ⬜ Partial | ~90 occurrences so far; most code still uses string concatenation |

### Phase 1B — Test infrastructure

| Task | Status | Notes |
|------|--------|-------|
| `tests/analysis/` suite | ✅ Done | `bad8583` — 7 test files: `test_config`, `test_loaders`, `test_metrics`, `test_plotting`, `test_processing`, `test_synchronization`, `test_cli` |
| Meaningful assertions in `test_python_magnetrun.py` | ⬜ Pending | File has 18 lines and **0 assertions** |
| Unit tests for `magnetdata.py` |  ✅ Done | `fromtdms`, `fromtxt`, `getData`, column renaming |
| Unit tests for `processing/` |  ✅ Done | `smoothers`, `trends`, `peaks`, `stats` — pure functions |
| Integration smoke-tests for CLI entry points | ✅ Done | tests/test_cli_entrypoints.py added |
| CI pipeline (GitHub Actions) | ⬜ Pending | No `.github/workflows/` directory yet |
| `mypy` pre-commit hook | ⬜ Pending | Present in `.pre-commit-config.yaml` but commented out |

### Phase 1C — Deprecate dead code

| Task | Status | Notes |
|------|--------|-------|
| Remove / formally deprecate `prepareData_legacy()` | ⬜ Partial | Marked deprecated with warning in `MagnetRun.py:32`, but still actively called in `utils/txt2csv.py` and `MagnetRun.py:229,268` |
| Remove placeholder CLI code | ⬜ Pending | `requests/cli.py` |
| Clean up WIP example scripts | ⬜ Pending | |

---

## Priority 2 — Unified Multi-Source Plotting

### Phase 2A — Unified data interface

| Task | Status | Notes |
|------|--------|-------|
| `DataProvider` protocol defined | ✅ Done | `hybrid/data_protocol.py` — `DataLoader` protocol with `getData`, `getKeys`, `getType`, `getTimeBase` |
| `MagnetRun` satisfies the protocol | ⬜ Pending | Not yet verified / enforced |
| `HybridRun` satisfies the protocol | ⬜ Pending | Not yet verified / enforced |

### Phase 2B — Time alignment layer

| Task | Status | Notes |
|------|--------|-------|
| Expose `get_time_range()` on `MagnetData` | ⬜ Pending | |
| Convert kHz/RMS seconds-from-day to UTC timestamp | ⬜ Pending | |
| `align_to_common_time(sources)` utility | ⬜ Pending | |

### Phase 2C — Extend `plot_data()` for hybrid

| Task | Status | Notes |
|------|--------|-------|
| `df_hybrid` + `hybrid_channels` params in `plot_data()` | ⬜ Pending | Current signature in `analysis/plotting.py:303` has no hybrid params |
| Hybrid data plotted on shared axes | ⬜ Pending | |
| Example: pupitre + pigbrother + hybrid on one graph | ✅ Done | `plot_hybrid_with_pupitre_tdms.py` (`ba4b59f`) |

### Phase 2D — Side-by-side comparison view

| Task | Status | Notes |
|------|--------|-------|
| `plot_comparison()` accepts `list[DataProvider]` | ⬜ Pending | Exists at `analysis/plotting.py:588` but does not take a DataProvider list |
| Auto-generate subplot grid (source × channel) | ⬜ Pending | |
| Shared / linked time axis | ⬜ Pending | |

### Phase 2E — Channel auto-mapping

| Task | Status | Notes |
|------|--------|-------|
| `CHANNEL_ALIASES` registry in `analysis/config.py` | ⬜ Pending | Not present |
| Fuzzy fallback for unmapped channels | ⬜ Pending | |

---

## Priority 3 — Code Readability & Maintenance

### Phase 3A — Break up monoliths

| File | Status | Notes |
|------|--------|-------|
| `magnetdata.py` (~1500 lines) | ⬜ Pending | Split into `io.py`, `transform.py`, `query.py` |
| `python_magnetrun.py` (~1300 lines) | ⬜ Pending | Split into `cli.py`, `commands/plot.py`, etc. |

### Phase 3B — Type hints

| Task | Status |
|------|--------|
| 100% type hints on new/modified code | ✅ Ongoing — enforced by ruff |
| Backfill `magnetdata.py` public API | ⬜ Pending |
| Backfill `MagnetRun.py` public API | ⬜ Pending |
| Backfill `analysis/plotting.py` | ⬜ Pending |
| Enable `mypy` in CI | ⬜ Pending |

### Phase 3C — Centralize configuration

| Task | Status | Notes |
|------|--------|-------|
| Move magic numbers to `analysis/config.py` | ⬜ Partial | `AnalysisConfig`, `ChannelMapping`, `SiteConfig` classes exist but not all constants centralised |
| `pydantic.BaseSettings` for site configs (M8, M9, M10) | ⬜ Pending | |

### Phase 3D — Tooling

| Task | Status | Notes |
|------|--------|-------|
| `ruff` pre-commit hook | ✅ Done | |
| `mypy` pre-commit hook | ⬜ Pending | Commented out in `.pre-commit-config.yaml` |

---

## Separation of `python_magnetcooling`

| Task | Status | Notes |
|------|--------|-------|
| Move cooling code to submodule | ✅ Done | `python_magnetcooling/` is a separate package |
| Core modules: `waterflow`, `cooling`, `thermohydraulics`, `channel`, … | ✅ Done | |
| `waterflow_factory` using magnetrun data | ⬜ Pending | See `python_magnetcooling/TODOs.md` |
| Check `thermohydraulics` — especially `gradHZH` case with P0 | ⬜ Pending | |
| Hysteresis model params for secondary flow | ⬜ Pending | |
| Cross-check feelpp implementation | ⬜ Pending | |
| Add option to pull data from magnetrun for waterflow pipeline | ⬜ Pending | |
| Add argparse to waterflow pipeline examples | ⬜ Pending | |
| Rerun validation against known cases | ⬜ Pending | |

---

## Quick Summary — What Remains

### Highest priority (do first)
1. **CI pipeline** — add `.github/workflows/ci.yml` running `ruff` + `pytest`
2. **Add assertions** to `tests/test_python_magnetrun.py` (currently 0)
3. **Enable `mypy`** pre-commit hook
4. **Finish `prepareData_legacy` removal** — update `utils/txt2csv.py` to not call it
5. **`print()` → `logger.*`** migration (1477 occurrences)

### Medium priority
6. **`DataProvider` protocol enforcement** — verify `MagnetRun` and `HybridRun` satisfy it
7. **Time alignment layer** — prerequisite for full unified plotting
8. **Extend `plot_data()`** with `df_hybrid` / `hybrid_channels` params
9. **`CHANNEL_ALIASES` registry** in `analysis/config.py`

### Lower priority / ongoing
10. Break up `magnetdata.py` and `python_magnetrun.py`
11. Backfill type hints on public APIs
12. `pathlib.Path` migration (opportunistic, per-file-touched)
13. `pydantic.BaseSettings` for site configs
