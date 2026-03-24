# Development Roadmap — python_magnetrun

*Created: 2026-03-23*

Priorities, in order:
1. **Correctness & reliability** — the package must work properly
2. **Unified plotting** — pupitre vs pigbrother vs hybrid data on the same graph or side by side
3. **Code readability & maintenance**

---

## Priority 1 — Package Correctness & Reliability

### Phase 1A: Stop the bleeding (1–2 weeks)

| Task | File(s) | Why |
|------|---------|-----|
| Fix 5 bare `except:` clauses | `utils/plots.py`, `utils/txt2csv.py`, `requests/cli.py`, `hybrid/trigger/plot_trigger_data.py`, `hybrid/vprocess/test.py` | Silently swallows `SystemExit` / `KeyboardInterrupt`, hides real bugs |
| Add file-format validation before parsing | `analysis/loaders.py`, `magnetdata.py` | Prevents confusing errors deep in TDMS/TXT parsing |
| Replace `print()` with `logger.*` | Whole codebase (1466 occurrences) | Cannot suppress output in library usage; masks real log messages |
| Migrate file paths to `pathlib.Path` | All modules using string path concatenation | Prevents path bugs and OS-specific issues |

### Phase 1B: Test infrastructure (2–4 weeks)

| Task | Details |
|------|---------|
| Add meaningful assertions to `tests/test_python_magnetrun.py` | Currently 0 assertions — completely hollow |
| Unit tests for `magnetdata.py` | Cover `fromtdms`, `fromtxt`, `getData`, column renaming |
| Unit tests for `processing/` modules | `smoothers`, `trends`, `peaks`, `stats` — pure functions, easy to test |
| Integration test for each CLI entry point | Smoke-test that each exits 0 with sample data |
| Add CI pipeline (GitHub Actions) | Run `ruff` + `pytest` on every push; fail on regression |

### Phase 1C: Deprecate dead code (1 week)

| Task | File(s) |
|------|---------|
| Remove or formally deprecate `prepareData_legacy()` | `MagnetRun.py` |
| Remove placeholder CLI code | `requests/cli.py` ("Replace this message...") |
| Clean up WIP example scripts or clearly mark as WIP | `examples/` |

---

## Priority 2 — Unified Multi-Source Plotting

**Current state:** `analysis/plotting.plot_data()` already overlays Overview + Archive + Pupitre
on shared axes with downsampling and incident markers.
**Gap:** Hybrid data (kHz/RMS) is completely excluded from this infrastructure.

### Phase 2A: Unified data interface (2–3 weeks)

A `DataProvider` protocol is already sketched in `hybrid/hybrid_run.py` but not enforced.
Formalize it as the common contract for all data sources:

```python
# python_magnetrun/protocol.py
from typing import Protocol
import pandas as pd
import numpy as np

class DataProvider(Protocol):
    def getData(self, key: str, downsample: int | None = None) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]: ...
    def getKeys(self) -> list[str]: ...
    def getType(self) -> int: ...         # 0=pupitre, 1=tdms_overview, 2=tdms_archive, 3=hybrid
    def getTimeBase(self) -> str: ...     # "timestamp" | "seconds_from_day" | "index"
```

Verify that `MagnetRun` and `HybridRun` both satisfy the protocol.
This is the critical dependency — all subsequent Phase 2 work builds on it.

### Phase 2B: Time alignment layer (1–2 weeks)

| Problem | Location | Fix |
|---------|----------|-----|
| TDMS uses `wf_start_time` metadata | `magnetdata.py:1311–1317` | Expose via common `get_time_range()` method |
| kHz/RMS uses seconds-from-day-start | `hybrid/hybrid_data.py` | Convert to UTC timestamp on load |
| Pupitre uses pandas `DatetimeIndex` | `magnetdata.py` | Already correct; use as reference time base |

Deliver a utility: `align_to_common_time(sources: list[DataProvider]) -> dict[str, pd.DataFrame]`
that returns all sources resampled/interpolated to a shared time axis.

### Phase 2C: Extend `plot_data()` to include hybrid (2–3 weeks)

Extend `analysis/plotting.plot_data()` with hybrid support:

```python
def plot_data(
    ...
    df_hybrid: HybridRun | None = None,          # NEW
    hybrid_channels: list[str] | None = None,    # NEW — logical channel names
    ...
)
```

Internally:
- Detect `getType() == 3`
- Call `getData(key, downsample=N)` and unpack `(array, time)` tuple
- Apply time alignment from Phase 2B
- Plot on shared axes with consistent color scheme (`PlotColors`)

### Phase 2D: Side-by-side comparison view (1–2 weeks)

Extend the existing `plot_comparison()` in `analysis/plotting.py` to accept
a list of `DataProvider` objects and auto-generate a grid of subplots:
- One column per source
- Shared time axis (linked zoom/pan)
- Same logical channel per row

```python
def plot_comparison(
    sources: list[DataProvider],
    channels: list[str],           # logical channel names
    title: str = "",
    figsize: tuple | None = None,
) -> plt.Figure: ...
```

### Phase 2E: Channel auto-mapping (1–2 weeks)

Currently callers must pass `channels_dict` and `pupitre_dict` manually.
Build a registry in `analysis/config.py`:

```python
CHANNEL_ALIASES: dict[str, list[str]] = {
    "IH":     ["Courant_GR1", "I_H1", "I_GR1"],
    "FlowH":  ["Debit_GR1", "flow_GR1"],
    "TinH":   ["Tin_GR1", "T_in_H1"],
    # ...
}
```

Auto-discover mappings at load time so callers only specify the logical channel name.
Add fuzzy fallback for channels not yet in the registry.

---

## Priority 3 — Code Readability & Maintenance

### Phase 3A: Break up the monoliths (3–4 weeks)

| File | Current size | Proposed split |
|------|-------------|----------------|
| `magnetdata.py` | 1500 lines, 50+ methods | `magnetdata/io.py` (TDMS/TXT loading), `magnetdata/transform.py` (column ops, units), `magnetdata/query.py` (getData, getStats, getKeys) |
| `python_magnetrun.py` | 1300 lines | `cli.py` (argparse only), `commands/plot.py`, `commands/stats.py`, `commands/convert.py`, etc. |

Public API must remain unchanged — refactor is internal only.

### Phase 3B: Complete type hints (ongoing)

- **Immediately:** 100% type hints on all new/modified code
- **Backfill:** public APIs of `magnetdata.py`, `MagnetRun.py`, `analysis/plotting.py`
- **Tooling:** enable `mypy` in CI starting from `analysis/` (already cleanest), expand incrementally
- Use `|` union syntax (PEP 604) consistently; remove `Optional[X]` and `Union[X, Y]`

### Phase 3C: Centralize configuration (1–2 weeks)

- Move magic numbers (energy balance thresholds, flow params, sampling rates)
  into `analysis/config.py` with named constants
- Consider `pydantic.BaseSettings` for site-specific config (M8, M9, M10)
  to get validation and documentation for free

### Phase 3D: Add `ruff` + `mypy` pre-commit hooks (1 day)

```toml
# pyproject.toml additions

[tool.ruff]
select = ["E", "F", "UP", "B", "SIM", "I"]

[tool.mypy]
python_version = "3.11"
disallow_untyped_defs = true
# Start lenient; tighten over time:
# strict = true
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
```

---

## Suggested Timeline

```
Month 1    │ Phase 1A + 1C  — bare excepts, print→logging, dead code removal
Month 2    │ Phase 1B        — test infrastructure, CI pipeline
Month 3    │ Phase 2A + 2B  — DataProvider protocol, time alignment layer
Month 4    │ Phase 2C + 2D  — extend plot_data(), side-by-side comparison
Month 5    │ Phase 2E + 3A  — channel auto-mapping, split monoliths
Month 6+   │ Phase 3B + 3C + 3D — type hints, config, pre-commit hooks
```

**Parallelization notes:**
- Phases 1A and 1C can be split across contributors immediately
- Phase 2A (DataProvider protocol) is the critical path blocker for all of Phase 2
- Phase 3B (type hints) can run in parallel with anything else as a background effort

---

## Quick Wins (do these first, any contributor)

1. Fix the 5 bare `except:` clauses — 30 minutes of work, immediate reliability gain
2. Add `ruff` pre-commit hook — 1 hour, enforces consistency from day one
3. Add assertions to `test_python_magnetrun.py` — makes CI meaningful immediately
4. Remove the placeholder text in `requests/cli.py`
5. Add `pathlib.Path` to any file you touch (opportunistic migration)
