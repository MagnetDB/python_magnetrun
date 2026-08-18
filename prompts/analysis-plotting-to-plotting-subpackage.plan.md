# Plan: migrate `analysis/plotting.py` → `plotting/` subpackage

## Goal

Make the `analysis` subpackage use the `plotting` subpackage for all
visualisation, removing `python_magnetrun/analysis/plotting.py` as the
authoritative source.

---

## Current state

| Location | What it contains |
|---|---|
| `analysis/plotting.py` | Downsampling helpers + magnetrun-specific plot functions |
| `plotting/` subpackage | Backend-agnostic `plot_subplots` / `plot_overlay`, style, annotations |
| `utils/downsampling.py` | Production-quality `DownsampleConfig` / `downsample_dataframe` |

`analysis/cli.py` imports `estimate_downsample_percent` and `plot_data`
from `.plotting` (i.e. `analysis/plotting.py`).

`analysis/__init__.py` re-exports everything from `analysis/plotting.py`.

---

## Proposed changes

### Step 1 — Create `plotting/magnetrun.py`

Move **all** functions from `analysis/plotting.py` here verbatim:

- Downsampling helpers  
  `downsample_for_plot`, `downsample_dataframe` (simple stride version),  
  `downsample_minmax`, `estimate_downsample_percent`
- Main plot functions  
  `plot_data`, `plot_comparison`, `plot_regimes`,  
  `plot_incidents_markers`, `plot_time_series`
- Matplotlib utilities  
  `setup_matplotlib_defaults`, `create_figure_grid`, `save_figure`

The module keeps its own `__all__` list and re-exports `PlotStyle`,
`PlotColors`, `DEFAULT_STYLE`, `DEFAULT_COLORS` from `.style`.

### Step 2 — Update `plotting/__init__.py`

Add imports from `.magnetrun` so every symbol is reachable via
`python_magnetrun.plotting`:

```python
from .magnetrun import (
    downsample_for_plot,
    downsample_dataframe as downsample_dataframe_simple,
    downsample_minmax,
    estimate_downsample_percent,
    plot_data,
    plot_comparison,
    plot_regimes,
    plot_incidents_markers,
    plot_time_series,
    setup_matplotlib_defaults,
    create_figure_grid,
    save_figure,
)
```

> **Note on name collision**: `utils/downsampling.py` already exports a
> more powerful `downsample_dataframe(df, time_col, value_cols, config)`.
> The simple stride version in `analysis/plotting.py` has a different
> signature `(df, percent, preserve_columns)`.  Export it under the same
> name since callers in `analysis/__init__.py` expect
> `downsample_dataframe`; the `plotting` subpackage exposes the same name
> pointing to the same simple implementation.  No callers use both.

### Step 3 — Replace `analysis/plotting.py` with re-exports

Keep the file so existing imports (`analysis/__init__.py`,
`tests/analysis/test_plotting.py`) keep working without change:

```python
# analysis/plotting.py  — thin re-export shim
from python_magnetrun.plotting.magnetrun import *   # noqa: F401,F403
from python_magnetrun.plotting.magnetrun import __all__
```

### Step 4 — Update `analysis/cli.py`

Change the local import inside `main()`:

```python
# Before
from .plotting import estimate_downsample_percent, plot_data

# After
from python_magnetrun.plotting import estimate_downsample_percent, plot_data
```

### Step 5 — Update `analysis/__init__.py`

Replace the import block that currently reads:

```python
from .plotting import (
    DEFAULT_COLORS, DEFAULT_STYLE, PlotColors, PlotStyle,
    create_figure_grid, downsample_dataframe, downsample_for_plot,
    downsample_minmax, estimate_downsample_percent, plot_comparison,
    plot_data, plot_incidents_markers, plot_regimes, plot_time_series,
    save_figure, setup_matplotlib_defaults,
)
```

with:

```python
from python_magnetrun.plotting import (
    DEFAULT_COLORS, DEFAULT_STYLE, PlotColors, PlotStyle,
    create_figure_grid, downsample_dataframe, downsample_for_plot,
    downsample_minmax, estimate_downsample_percent, plot_comparison,
    plot_data, plot_incidents_markers, plot_regimes, plot_time_series,
    save_figure, setup_matplotlib_defaults,
)
```

---

## Files changed

| File | Action |
|---|---|
| `python_magnetrun/plotting/magnetrun.py` | **Create** |
| `python_magnetrun/plotting/__init__.py` | **Update** (add magnetrun imports) |
| `python_magnetrun/analysis/plotting.py` | **Replace** with re-export shim |
| `python_magnetrun/analysis/cli.py` | **Update** import path |
| `python_magnetrun/analysis/__init__.py` | **Update** import path |

Tests in `tests/analysis/test_plotting.py` continue to import from
`python_magnetrun.analysis.plotting` and keep working via the re-export
shim — **no test changes needed**.

---

## Out of scope

- Merging the simple `downsample_dataframe` with the `DownsampleConfig`-
  based one in `utils/downsampling.py` — a separate refactor.
- Moving `plot_tlcc` / `plot_dtw_alignment` from `analysis/metrics.py`
  to `plotting/` — separate task.
