# `python_magnetrun.plotting`

Backend-agnostic plotting subpackage for magnetrun data.

---

## Public API

| Symbol | Description |
|--------|-------------|
| `PlotStyle` | Dataclass for figure size, DPI, grid, fonts |
| `PlotColors` | Dataclass for per-source and regime colours |
| `get_backend(name)` | Factory returning a `PlottingBackend` instance |
| `plot_subplots(data, fields, …)` | N fields as stacked subplots sharing a time axis |
| `plot_overlay(data, fields, …)` | N fields on one axes, with optional per-series normalisation |
| `AnnotationManager` | Backend-agnostic clickable incident/event annotations |

---

## Backend selection

| Goal | `backend=` |
|------|-----------|
| Static PNG / SVG file | `"matplotlib"` (default) |
| JSON for JS frontend / REST API | `"plotly"` |
| Jupyter / marimo notebook | `"plotly"` |
| voilà / voici dashboard | `"plotly"` |
| Dash app with huge datasets | `"plotly-resampler"` |
| Jupyter / marimo ipywidget | `"plotly-widget"` |

`"plotly-resampler"` and `"plotly-widget"` require a **live Python kernel** and
the `plotly-resampler` package.  They perform view-dependent on-the-fly
aggregation (MinMaxLTTB) on every pan/zoom, so the `DownsampleConfig`
pre-computation step is skipped automatically.

---

## Quick-start

```python
import pandas as pd
from python_magnetrun.plotting import plot_subplots, plot_overlay, get_backend

# Stacked subplots — matplotlib (static PNG)
fig = plot_subplots(df, ["Courant_GR1", "Field"], t_col="t")
get_backend().save(fig, "run.png")

# Overlay with normalisation — plotly (interactive)
fig = plot_overlay(
    df, ["Courant_GR1", "Field"],
    normalize=True,
    units={"Courant_GR1": "A", "Field": "T"},
    backend="plotly",
)
get_backend("plotly").show(fig)

# JSON export for a REST endpoint
spec = get_backend("plotly").to_json(fig)  # application/json
```

### JS frontend integration

```
GET /api/plot?fields=Field,Courant_GR1&backend=plotly
  ──► load data  (MagnetData / HybridRun)
  ──► plot_subplots(data, fields, backend="plotly")
  ──► return backend.to_json(fig)          # application/json

JS:
  fetch("/api/plot?…")
  .then(spec => Plotly.react("div-id", spec.data, spec.layout))
```

### marimo / voilà dashboards

```python
import marimo as mo
from python_magnetrun.plotting import plot_subplots

fig = plot_subplots(df, ["Field"], backend="plotly")
mo.ui.plotly(fig)   # native marimo widget — interactive out of the box
```

### Dynamic resampling (very large datasets)

```python
# Dash — FigureResampler, live kernel required
fig = plot_subplots(df, ["Field"], backend="plotly-resampler")
get_backend("plotly-resampler").show(fig)   # fig.show_dash()

# Jupyter / marimo ipywidget — FigureWidgetResampler
fig = plot_subplots(df, ["Field"], backend="plotly-widget")
```

## Per-series style (`field_styles`)

`plot_subplots`, `plot_overlay`, and `plot_xy` accept a `field_styles` parameter
— a list of `(linestyle, marker, markevery, alpha)` tuples, one per field (extra
entries are ignored, missing entries fall back to defaults).

| Position | Type | Meaning |
|----------|------|---------|
| 0 | `str \| None` | Line style (`"-"`, `"--"`, `"-."`, `"none"`) |
| 1 | `str \| None` | Matplotlib/Plotly marker (e.g. `"o"`, `"s"`, `"D"`) |
| 2 | `int \| None` | `markevery` — draw marker every *N* points |
| 3 | `float \| None` | `alpha` — opacity in `[0, 1]` |

```python
from python_magnetrun.plotting import plot_overlay

fig = plot_overlay(
    df,
    ["Référence_A1", "Référence_A2"],
    field_styles=[
        ("-",    None, None, None),   # A1: solid line, full opacity
        ("--",   "o",  10,   0.5),    # A2: dashed line + circles every 10 pts, 50% opacity
    ],
    backend="matplotlib",
)
```

From the CLI the same options are expressed via `--field_style FIELD=STYLESPEC`
where the syntax is `[LINESTYLE][MARKER][:N][@ALPHA]` (see project README).

---



Pass a `DownsampleConfig` to pre-reduce data before plotting (useful for
matplotlib or static Plotly export):

```python
from python_magnetrun.utils.downsampling import DownsampleConfig
from python_magnetrun.plotting import plot_subplots

cfg = DownsampleConfig(n_out=10_000, method="minmax_lttb")
fig = plot_subplots(df, ["Field"], downsample=cfg, backend="plotly")
```

Pre-downsampling is **skipped automatically** when using `"plotly-resampler"`
or `"plotly-widget"` — tier-3 dynamic resampling takes over.

Three-tier strategy:

```
Tier 1 — data loading    →  DownsampleConfig (any backend, static-safe)
Tier 2 — plot creation   →  DownsampleConfig pre-computed (static / REST / matplotlib)
Tier 3 — user interaction →  FigureResampler  (live kernel: Jupyter / voilà / marimo / Dash)
```

---

## Annotations

```python
from python_magnetrun.plotting import AnnotationManager, get_backend
from python_magnetrun.plotting.matplotlib_backend import MatplotlibBackend

b = MatplotlibBackend()
fig = b.subplots(1)

mgr = AnnotationManager(b)
mgr.add(fig, ax_idx=0, t=42.0, label="spike #1", detail={"idx": 0, "df": idf, …})
mgr.connect(fig)   # wires matplotlib pick-event; no-op for plotly
```

---

## Optional dependencies

| Feature | Install |
|---------|---------|
| Interactive Plotly + static image export | `pip install python_magnetrun[plotting]` |
| Dynamic view-dependent resampling | `pip install python_magnetrun[resampler]` |
| Advanced downsampling (MinMaxLTTB) | `pip install python_magnetrun[hybrid]` |

All optional packages are imported lazily with `try/except ImportError`, so
the subpackage is fully importable without them.

---

## Module layout

```
plotting/
├── __init__.py                    # public re-exports
├── backend.py                     # PlottingBackend Protocol + get_backend()
├── matplotlib_backend.py          # MatplotlibBackend
├── plotly_backend.py              # PlotlyBackend — static + to_json()
├── plotly_resampler_backend.py    # PlotlyResamplerBackend — live kernel only
├── style.py                       # PlotStyle, PlotColors
├── timeseries.py                  # plot_subplots(), plot_overlay()
└── annotations.py                 # AnnotationManager
```
