# Prompt: Phase 4 — Interactive Dashboards

## Context

`python_magnetrun` currently has two non-importable scripts in `panels/`. This prompt
covers **Phase 4** of the improvement plan: replacing those scripts with a proper
`dashboards/` subpackage built on **Plotly + Dash + plotly-resampler**.

**Why Plotly/Dash instead of Panel/hvplot?**
Magnet run data spans two very different scale regimes:
- **Pupitre data** — ~1 Hz sampling, tens of thousands of rows → any stack works.
- **FEPC hybrid data** — 10 kHz sampling, 10 M+ rows per acquisition → requires
  *dynamic, viewport-aware downsampling*. `plotly-resampler` applies MinMaxLTTB
  server-side on every pan/zoom so the browser only ever receives ~1 000 points,
  regardless of the underlying dataset size. Panel/hvplot have no equivalent mechanism
  without bespoke pre-aggregation.

`plotly-resampler` provides two integration classes:
- **`FigureResampler`** — wraps a Plotly figure and runs a Dash server; resampling
  happens via Dash callbacks. Used by the `magnetrun-dashboard` CLI.
- **`FigureWidgetResampler`** — uses IPython widget events; works directly in Jupyter
  notebooks without a separate server process.

**Prerequisite:** Phases 1, 2, 3, and 3b must be complete.

Reference document: `IMPROVEMENT_PLAN.md` §Phase 4.

---

## Objective

1. Create `python_magnetrun/dashboards/` as an importable, testable subpackage.
2. Implement `run_overview`, `field_analysis`, `comparison`, and `hybrid_monitor`
   dashboards using `FigureResampler` (Dash) and `FigureWidgetResampler` (notebook).
3. Add a `magnetrun-dashboard` CLI that serves dashboards in a browser.
4. Add `magnetrun-to-notebook` CLI that generates pre-filled Jupyter notebooks.
5. (Optional) Wire the comparison dashboard to the `MagnetAPIClient` for multi-run
   loading from the database.

---

## Prerequisites — Add dependencies

```toml
[project.optional-dependencies]
dashboard = [
    "plotly>=5.22",
    "dash>=2.17",
    "plotly-resampler>=0.10",   # dynamic resampling — core of Phase 4
    "tsdownsample>=0.1.3",      # MinMaxLTTB backend for plotly-resampler
]
notebook = [
    "nbformat>=5.10",
    "jupyter_client>=8.0",
    "plotly-resampler>=0.10",   # FigureWidgetResampler for notebook use
]
```

Install for development:
```bash
pip install -e ".[dashboard,notebook,dev]"
```

---

## Task 4.1 — Restructure `panels/` into `dashboards/` subpackage

**Steps:**

1. Create the new directory structure:
   ```
   python_magnetrun/dashboards/
   ├── __init__.py
   ├── run_overview.py
   ├── field_analysis.py
   ├── comparison.py
   ├── hybrid_monitor.py
   ├── widgets.py
   └── cli.py
   ```

2. Move useful code from existing panel scripts:
   ```bash
   # Read both existing files first, then port relevant logic into the new modules
   cat python_magnetrun/panels/panel-mrecord.py
   cat python_magnetrun/panels/panel-mrecord-vs-time.py
   ```

3. Leave `panels/` in place with a deprecation notice in each script's docstring.
   Do not delete until v0.4.0.

---

## Task 4.2 — Shared figure helpers in `dashboards/figures.py`

These helpers produce `plotly.graph_objects.Figure` objects that are then wrapped in
either `FigureResampler` (Dash) or `FigureWidgetResampler` (notebook).
The resampling wrapper is applied by the dashboard layer, not here — keeping figure
construction testable without a Dash server.

```python
"""
Shared Plotly figure builders for python_magnetrun dashboards.

Each function returns a plain plotly.graph_objects.Figure.
The caller wraps it in FigureResampler or FigureWidgetResampler as needed.

Usage::

    from plotly_resampler import FigureResampler
    from python_magnetrun.dashboards.figures import make_time_series_figure

    fig = make_time_series_figure(df, y_cols=["IH", "IB"], title="Currents")
    fig_r = FigureResampler(fig)        # Dash mode
    fig_r.show_dash(mode="inline")      # or: pn.serve(fig_r) with panel integration
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    import pandas as pd


def make_time_series_figure(
    df: "pd.DataFrame",
    y_cols: list[str],
    time_col: str = "t",
    title: str = "",
    y_label: str = "",
    colors: list[str] | None = None,
) -> go.Figure:
    """
    Build a multi-trace time-series figure.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    y_cols : list[str]
        Column names to plot as separate traces.
    time_col : str
        Name of the time (x) column.
    title : str
        Figure title.
    y_label : str
        Y-axis label.
    colors : list[str], optional
        Per-trace colour strings. If shorter than y_cols, cycles.

    Returns
    -------
    go.Figure
        Plain Plotly figure — wrap in FigureResampler before serving.

    Notes
    -----
    Use ``go.Scattergl`` (WebGL) for traces with >10 000 points rendered
    statically; ``go.Scatter`` otherwise. When wrapped in FigureResampler
    the downsampling removes the need for WebGL at the browser level.
    """
    fig = go.Figure()
    _colors = colors or []
    for i, col in enumerate(y_cols):
        if col not in df.columns:
            continue
        color = _colors[i % len(_colors)] if _colors else None
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[col],
            name=col,
            mode="lines",
            line=dict(color=color) if color else {},
        ))
    fig.update_layout(
        title=title,
        xaxis_title=time_col,
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=60),
        height=320,
    )
    return fig


def make_scatter_figure(
    df: "pd.DataFrame",
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str = "",
) -> go.Figure:
    """
    Build a scatter figure (B vs I, etc.), optionally coloured by a third column.

    Returns
    -------
    go.Figure
        Plain Plotly figure.
    """
    marker: dict = {}
    if color_col and color_col in df.columns:
        marker = dict(color=df[color_col], colorscale="Viridis",
                      showscale=True, colorbar=dict(title=color_col))

    fig = go.Figure(go.Scatter(
        x=df.get(x_col, []),
        y=df.get(y_col, []),
        mode="markers",
        marker=marker or dict(size=3),
        name=f"{y_col} vs {x_col}",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=380,
    )
    return fig
```

---

## Task 4.3 — `run_overview` dashboard

**File:** `python_magnetrun/dashboards/run_overview.py`

This is the primary dashboard. It uses `FigureResampler` so that Pupitre runs with
tens of thousands of rows and FEPC runs with millions of rows are handled identically.

```python
"""
Run overview dashboard — Plotly + plotly-resampler.

Shows field, currents, flow, and temperature vs time.
Resampling is dynamic: only the points visible in the current viewport
are sent to the browser (MinMaxLTTB algorithm).

Usage — Dash web app (magnetrun-dashboard CLI)::

    from python_magnetrun import MagnetRun
    from python_magnetrun.dashboards.run_overview import run_overview_app

    run = MagnetRun.fromtxt("run_20240315.txt")
    app = run_overview_app(run)
    app.run(debug=False, port=5006)

Usage — Jupyter notebook::

    from python_magnetrun.dashboards.run_overview import run_overview_widget
    fig = run_overview_widget(run)   # returns FigureWidgetResampler; display inline
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from python_magnetrun import MagnetRun

logger = logging.getLogger(__name__)

_DEFAULT_CURRENT_KEYS = ["IH", "IB"]
_DEFAULT_FLOW_KEYS    = ["FlowH", "FlowB"]
_DEFAULT_TEMP_KEYS    = ["teb", "tsb"]


def _build_overview_traces(run: "MagnetRun") -> go.Figure:
    """
    Build a multi-subplot Plotly figure with one subplot per channel group.

    Returns a plain go.Figure — caller wraps in FigureResampler or
    FigureWidgetResampler.
    """
    from plotly.subplots import make_subplots

    df = run.getData()
    time_col = "t" if "t" in df.columns else df.columns[0]

    subplot_titles = ["Field (T)", "Currents (A)", "Flow (l/s)", "Temperature (°C)"]
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
    )
    fig.update_layout(
        height=900,
        margin=dict(l=60, r=20, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # Row 1 — Field
    if "Field" in df.columns:
        fig.add_trace(
            go.Scatter(x=df[time_col], y=df["Field"], name="Field", mode="lines",
                       line=dict(color="navy")),
            row=1, col=1,
        )

    # Row 2 — Currents
    colors = ["royalblue", "tomato"]
    for i, key in enumerate([k for k in _DEFAULT_CURRENT_KEYS if k in df.columns]):
        fig.add_trace(
            go.Scatter(x=df[time_col], y=df[key], name=key, mode="lines",
                       line=dict(color=colors[i % len(colors)])),
            row=2, col=1,
        )

    # Row 3 — Flow rates
    colors = ["seagreen", "darkorange"]
    for i, key in enumerate([k for k in _DEFAULT_FLOW_KEYS if k in df.columns]):
        fig.add_trace(
            go.Scatter(x=df[time_col], y=df[key], name=key, mode="lines",
                       line=dict(color=colors[i % len(colors)])),
            row=3, col=1,
        )

    # Row 4 — Temperatures
    colors = ["orchid", "peru"]
    for i, key in enumerate([k for k in _DEFAULT_TEMP_KEYS if k in df.columns]):
        fig.add_trace(
            go.Scatter(x=df[time_col], y=df[key], name=key, mode="lines",
                       line=dict(color=colors[i % len(colors)])),
            row=4, col=1,
        )

    fig.update_xaxes(title_text=f"Time ({time_col})", row=4, col=1)
    return fig


def run_overview_app(run: "MagnetRun", port: int = 5006, debug: bool = False):
    """
    Build and return a Dash app with live resampling for a single run.

    The app is NOT started — call app.run() or pass it to the CLI.

    Parameters
    ----------
    run : MagnetRun
    port : int
    debug : bool

    Returns
    -------
    dash.Dash
        Configured Dash application with FigureResampler.
    """
    from plotly_resampler import FigureResampler
    import dash
    from dash import dcc, html
    from plotly_resampler.callbacks import construct_update_data

    fig = FigureResampler(_build_overview_traces(run))
    title = getattr(run.getMData(), "FileName", "Run Overview")

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2(title),
        dcc.Graph(id="overview-graph", figure=fig),
        dcc.Loading(dcc.Store(id="overview-store")),
    ])

    fig.register_update_graph_callback(app, "overview-graph", "overview-store")

    logger.info("Run overview app built for: %s", title)
    return app


def run_overview_widget(run: "MagnetRun"):
    """
    Return a FigureWidgetResampler for Jupyter notebook use.

    Display it directly in a cell: just evaluate the returned object
    as the last expression in a cell, or call ``display(fig)``.

    Parameters
    ----------
    run : MagnetRun

    Returns
    -------
    FigureWidgetResampler
    """
    from plotly_resampler import FigureWidgetResampler
    return FigureWidgetResampler(_build_overview_traces(run))
```

---

## Task 4.4 — `field_analysis` dashboard

**File:** `python_magnetrun/dashboards/field_analysis.py`

```python
"""
Field analysis dashboard: B vs I curves, hysteresis, piecewise linear fits.
"""
from __future__ import annotations

import panel as pn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python_magnetrun import MagnetRun


def field_analysis_dashboard(run: "MagnetRun") -> pn.viewable.Viewable:
    """
    Build a dashboard for field-vs-current analysis.

    Panels
    ------
    - B vs IH scatter (colored by time)
    - B vs IB scatter
    - Hysteresis loop (if multiple ramp up/down cycles)
    - Residuals from linear fit

    Parameters
    ----------
    run : MagnetRun

    Returns
    -------
    pn.viewable.Viewable
    """
    import hvplot.pandas  # noqa: F401
    pn.extension()

    df = run.getData()
    time_col = "t" if "t" in df.columns else df.columns[0]

    plots = []

    if "IH" in df.columns and "Field" in df.columns:
        plots.append(
            df.hvplot.scatter(
                x="IH", y="Field", c=time_col,
                title="Field vs IH (colored by time)",
                cmap="viridis",
                height=350, width=450,
            )
        )

    if "IB" in df.columns and "Field" in df.columns:
        plots.append(
            df.hvplot.scatter(
                x="IB", y="Field", c=time_col,
                title="Field vs IB (colored by time)",
                cmap="plasma",
                height=350, width=450,
            )
        )

    if not plots:
        return pn.pane.Str("Required columns (Field, IH or IB) not found in data.")

    return pn.Column(
        pn.pane.Markdown("## Field Analysis"),
        pn.Row(*plots),
    )
```

---

## Task 4.5 — `comparison` dashboard

**File:** `python_magnetrun/dashboards/comparison.py`

```python
"""
Multi-run comparison dashboard.

Overlays field profiles from multiple runs, time-normalized to run start,
and shows side-by-side statistics.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn

if TYPE_CHECKING:
    from python_magnetrun import MagnetRun

logger = logging.getLogger(__name__)


def comparison_dashboard(
    runs: list[tuple[str, "MagnetRun"]],
    normalize_time: bool = True,
) -> pn.viewable.Viewable:
    """
    Build a comparison dashboard for multiple runs.

    Parameters
    ----------
    runs : list of (label, MagnetRun) tuples
        List of labelled runs to compare.
    normalize_time : bool
        If True, shift each run's time axis so t=0 is the start of the run.

    Returns
    -------
    pn.viewable.Viewable

    Examples
    --------
    >>> from python_magnetrun import MagnetRun
    >>> from python_magnetrun.dashboards.comparison import comparison_dashboard
    >>> r1 = MagnetRun.fromtxt("run_A.txt")
    >>> r2 = MagnetRun.fromtxt("run_B.txt")
    >>> dashboard = comparison_dashboard([("Run A", r1), ("Run B", r2)])
    >>> dashboard.show()
    """
    import hvplot.pandas  # noqa: F401
    pn.extension()

    if not runs:
        return pn.pane.Str("No runs provided.")

    # Build combined DataFrame
    frames = []
    for label, run in runs:
        df = run.getData().copy()
        time_col = "t" if "t" in df.columns else df.columns[0]
        if normalize_time:
            df[time_col] = df[time_col] - df[time_col].min()
        df["_run"] = label
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    time_col = "t" if "t" in combined.columns else combined.columns[0]

    # --- Field overlay ---
    field_plots = pn.pane.Str("No 'Field' column available")
    if "Field" in combined.columns:
        field_plots = combined.hvplot.line(
            x=time_col, y="Field", by="_run",
            title="Field comparison (T)",
            height=350, width=900,
        )

    # --- Statistics table ---
    stats_rows = []
    for label, run in runs:
        df = run.getData()
        row = {"Run": label}
        for key in ["Field", "IH", "IB", "FlowH", "FlowB"]:
            if key in df.columns:
                row[f"{key}_max"] = round(df[key].max(), 3)
                row[f"{key}_mean"] = round(df[key].mean(), 3)
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_table = pn.pane.DataFrame(stats_df, index=False)

    # --- Run selector for detailed view ---
    run_labels = [label for label, _ in runs]
    run_select = pn.widgets.Select(name="Inspect run", options=run_labels)

    @pn.depends(run_select)
    def detail_plot(selected_label):
        run_dict = dict(runs)
        run = run_dict.get(selected_label)
        if run is None:
            return pn.pane.Str("Run not found")
        df = run.getData()
        tc = "t" if "t" in df.columns else df.columns[0]
        available = [k for k in ["IH", "IB", "FlowH", "FlowB"] if k in df.columns]
        if not available:
            return pn.pane.Str("No detail channels available")
        return df.hvplot.line(x=tc, y=available, title=f"Detail: {selected_label}",
                              height=300, width=900)

    return pn.Column(
        pn.pane.Markdown("## Run Comparison"),
        field_plots,
        pn.pane.Markdown("### Statistics"),
        stats_table,
        pn.pane.Markdown("### Detail view"),
        run_select,
        pn.panel(detail_plot),
    )


def comparison_dashboard_from_api(
    run_ids: list[int],
    client: "MagnetAPIClient | None" = None,
) -> pn.viewable.Viewable:
    """
    Load runs from the API and build a comparison dashboard.

    Parameters
    ----------
    run_ids : list[int]
        Run IDs to compare.
    client : MagnetAPIClient, optional
        API client. If None, instantiated from environment variables.

    Returns
    -------
    pn.viewable.Viewable
    """
    from python_magnetrun.api import MagnetAPIClient
    if client is None:
        client = MagnetAPIClient()

    runs = []
    for rid in run_ids:
        try:
            run = client.get_run_data(rid)
            runs.append((f"Run {rid}", run))
        except Exception as exc:
            logger.error("Failed to load run %d: %s", rid, exc)

    return comparison_dashboard(runs)
```

---

## Task 4.6 — `hybrid_monitor` dashboard

**File:** `python_magnetrun/dashboards/hybrid_monitor.py`

This is the dashboard where `plotly-resampler` matters most. FEPC data at 10 kHz
with acquisition windows of hundreds of seconds results in tens of millions of rows.
`FigureResampler` handles this natively: the full dataset stays on the server and only
the ~1 000 points relevant to the current viewport are sent to the browser.

```python
"""
Hybrid FEPC data monitor — Plotly + plotly-resampler.

The full kHz/RMS dataset is kept server-side. Dynamic MinMaxLTTB
aggregation is triggered on every pan or zoom via Dash callbacks.

Usage — Dash web app::

    from python_magnetrun.hybrid.hybrid_data import HybridData
    from python_magnetrun.dashboards.hybrid_monitor import hybrid_monitor_app

    hybrid = HybridData("/path/to/hybrid_dir/")
    app = hybrid_monitor_app(hybrid)
    app.run(port=5007)

Usage — Jupyter notebook::

    from python_magnetrun.dashboards.hybrid_monitor import hybrid_monitor_widget
    fig = hybrid_monitor_widget(hybrid, kHz_channel="U_coil1")
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from python_magnetrun.hybrid.hybrid_data import HybridData

logger = logging.getLogger(__name__)


def _build_hybrid_figure(
    hybrid: "HybridData",
    kHz_channel: str | None = None,
    rms_channel: str | None = None,
) -> go.Figure:
    """
    Build a two-subplot figure from hybrid FEPC data.

    Loads data from the HybridData object; does NOT downsample —
    the caller wraps in FigureResampler for dynamic aggregation.

    Parameters
    ----------
    hybrid : HybridData
    kHz_channel : str, optional
        kHz channel to show in the top subplot.
    rms_channel : str, optional
        RMS channel to show in the bottom subplot.

    Returns
    -------
    go.Figure
        Plain Plotly figure with raw (full-resolution) traces.
    """
    kHz_keys  = getattr(hybrid, "kHz_keys",  [])
    rms_keys  = getattr(hybrid, "rms_keys",  [])

    kHz_ch  = kHz_channel  or (kHz_keys[0]  if kHz_keys  else None)
    rms_ch  = rms_channel  or (rms_keys[0]  if rms_keys  else None)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"kHz: {kHz_ch}" if kHz_ch else "kHz (no channel selected)",
            f"RMS: {rms_ch}" if rms_ch else "RMS (no channel selected)",
        ],
        vertical_spacing=0.08,
    )
    fig.update_layout(height=700, margin=dict(l=60, r=20, t=80, b=60))

    if kHz_ch:
        try:
            df_kHz = hybrid.read_kHz(kHz_ch)
            t_col = df_kHz.columns[0]
            fig.add_trace(
                go.Scatter(
                    x=df_kHz[t_col],
                    y=df_kHz[kHz_ch],
                    name=kHz_ch,
                    mode="lines",
                    line=dict(color="steelblue", width=1),
                ),
                row=1, col=1,
            )
            logger.info("Loaded kHz channel %s: %d points", kHz_ch, len(df_kHz))
        except Exception as exc:
            logger.error("Failed to load kHz channel %s: %s", kHz_ch, exc)

    if rms_ch:
        try:
            df_rms = hybrid.read_rms(rms_ch)
            t_col = df_rms.columns[0]
            fig.add_trace(
                go.Scatter(
                    x=df_rms[t_col],
                    y=df_rms[rms_ch],
                    name=rms_ch,
                    mode="lines",
                    line=dict(color="darkorange", width=1),
                ),
                row=2, col=1,
            )
            logger.info("Loaded RMS channel %s: %d points", rms_ch, len(df_rms))
        except Exception as exc:
            logger.error("Failed to load RMS channel %s: %s", rms_ch, exc)

    return fig


def hybrid_monitor_app(hybrid: "HybridData"):
    """
    Build a Dash app for FEPC hybrid data with live viewport resampling.

    Dropdowns for kHz and RMS channel selection trigger a figure rebuild;
    FigureResampler handles dynamic aggregation on pan/zoom.

    Returns
    -------
    dash.Dash
    """
    from plotly_resampler import FigureResampler
    import dash
    from dash import dcc, html, Input, Output, callback

    kHz_keys = getattr(hybrid, "kHz_keys", []) or ["(none)"]
    rms_keys = getattr(hybrid, "rms_keys", []) or ["(none)"]

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("Hybrid FEPC Monitor"),
        html.Div([
            html.Label("kHz channel"),
            dcc.Dropdown(id="kHz-select", options=kHz_keys, value=kHz_keys[0]),
            html.Label("RMS channel"),
            dcc.Dropdown(id="rms-select", options=rms_keys, value=rms_keys[0]),
        ], style={"display": "flex", "gap": "24px", "alignItems": "center",
                  "marginBottom": "12px"}),
        dcc.Graph(id="hybrid-graph"),
        dcc.Store(id="hybrid-store"),
        dcc.Loading(html.Div(id="hybrid-loading")),
    ])

    # One FigureResampler instance per callback invocation.
    # Store in app.server._fr to allow the resampler's own callbacks to work.
    @app.callback(
        Output("hybrid-graph", "figure"),
        Output("hybrid-store", "data"),
        Input("kHz-select", "value"),
        Input("rms-select", "value"),
    )
    def update_figure(kHz_ch, rms_ch):
        plain_fig = _build_hybrid_figure(
            hybrid,
            kHz_channel=kHz_ch if kHz_ch != "(none)" else None,
            rms_channel=rms_ch if rms_ch != "(none)" else None,
        )
        fr = FigureResampler(plain_fig)
        # Register resampler so zoom/pan callbacks work
        fr.register_update_graph_callback(app, "hybrid-graph", "hybrid-store")
        return fr, {}

    return app


def hybrid_monitor_widget(
    hybrid: "HybridData",
    kHz_channel: str | None = None,
    rms_channel: str | None = None,
):
    """
    Return a FigureWidgetResampler for Jupyter notebook use.

    The full dataset is kept in memory; only viewport-visible points
    are rendered. Pan and zoom trigger automatic re-aggregation via
    IPython widget events — no Dash server needed.

    Parameters
    ----------
    hybrid : HybridData
    kHz_channel : str, optional
    rms_channel : str, optional

    Returns
    -------
    FigureWidgetResampler
    """
    from plotly_resampler import FigureWidgetResampler
    plain_fig = _build_hybrid_figure(hybrid, kHz_channel, rms_channel)
    return FigureWidgetResampler(plain_fig)
```

---

## Task 4.7 — `dashboards/__init__.py`

```python
"""
python_magnetrun interactive dashboards.

All dashboards require plotly, dash, and plotly-resampler.
Install with: pip install python-magnetrun[dashboard]

Two modes per dashboard
-----------------------
- ``*_app(run)``    → returns a Dash app; call app.run() or use the CLI.
- ``*_widget(run)`` → returns a FigureWidgetResampler for Jupyter notebooks.

Available dashboards
--------------------
- run_overview_app / run_overview_widget
- field_analysis_app / field_analysis_widget
- comparison_app / comparison_widget
- hybrid_monitor_app / hybrid_monitor_widget  ← critical for 10 M+ point FEPC data
"""
from __future__ import annotations


def _require_dashboard_deps() -> None:
    missing = []
    for pkg in ("plotly", "dash", "plotly_resampler"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Dashboards require: {', '.join(missing)}.\n"
            "Install with: pip install python-magnetrun[dashboard]"
        )


def run_overview_app(run, **kwargs):
    _require_dashboard_deps()
    from python_magnetrun.dashboards.run_overview import run_overview_app as _fn
    return _fn(run, **kwargs)


def run_overview_widget(run, **kwargs):
    _require_dashboard_deps()
    from python_magnetrun.dashboards.run_overview import run_overview_widget as _fn
    return _fn(run, **kwargs)


def field_analysis_app(run, **kwargs):
    _require_dashboard_deps()
    from python_magnetrun.dashboards.field_analysis import field_analysis_app as _fn
    return _fn(run, **kwargs)


def field_analysis_widget(run, **kwargs):
    _require_dashboard_deps()
    from python_magnetrun.dashboards.field_analysis import field_analysis_widget as _fn
    return _fn(run, **kwargs)


def comparison_app(runs, **kwargs):
    _require_dashboard_deps()
    from python_magnetrun.dashboards.comparison import comparison_app as _fn
    return _fn(runs, **kwargs)


def hybrid_monitor_app(hybrid, **kwargs):
    _require_dashboard_deps()
    from python_magnetrun.dashboards.hybrid_monitor import hybrid_monitor_app as _fn
    return _fn(hybrid, **kwargs)


def hybrid_monitor_widget(hybrid, **kwargs):
    _require_dashboard_deps()
    from python_magnetrun.dashboards.hybrid_monitor import hybrid_monitor_widget as _fn
    return _fn(hybrid, **kwargs)


__all__ = [
    "run_overview_app",
    "run_overview_widget",
    "field_analysis_app",
    "field_analysis_widget",
    "comparison_app",
    "hybrid_monitor_app",
    "hybrid_monitor_widget",
]
```

---

## Task 4.8 — `magnetrun-dashboard` CLI

**File:** `python_magnetrun/dashboards/cli.py`

```python
"""
CLI for serving python_magnetrun Dash dashboards with live resampling.

Usage::

    magnetrun-dashboard overview run_20240315.txt
    magnetrun-dashboard overview run_20240315.txt --port 8050 --debug
    magnetrun-dashboard hybrid  /path/to/hybrid_dir/
    magnetrun-dashboard hybrid  /path/to/hybrid_dir/ --kHz U_coil1 --rms U_rms1
    magnetrun-dashboard compare run_A.txt run_B.txt
    magnetrun-dashboard compare --run-ids 42 43
    magnetrun-dashboard to-notebook run_20240315.txt -o analysis.ipynb
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magnetrun-dashboard",
        description="Serve a python_magnetrun Dash dashboard with live resampling",
    )
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--debug", action="store_true",
                        help="Run Dash in debug mode (auto-reload)")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="dashboard", required=True)

    # overview
    p_ov = sub.add_parser("overview", help="Run time-series overview")
    p_ov.add_argument("file", help="Run data file (.txt, .tdms, .csv)")

    # field
    p_fi = sub.add_parser("field", help="Field vs current analysis")
    p_fi.add_argument("file")

    # compare
    p_cm = sub.add_parser("compare", help="Multi-run comparison")
    p_cm.add_argument("files", nargs="*", help="Run data files")
    p_cm.add_argument("--run-ids", nargs="+", type=int, metavar="ID",
                      help="Run IDs to load from the API")

    # hybrid — most important: handles 10 M+ point FEPC data
    p_hy = sub.add_parser("hybrid", help="FEPC hybrid data monitor (10 M+ pts)")
    p_hy.add_argument("directory", help="Hybrid FEPC data directory")
    p_hy.add_argument("--kHz", metavar="CHANNEL", default=None,
                      help="kHz channel to show on startup")
    p_hy.add_argument("--rms", metavar="CHANNEL", default=None,
                      help="RMS channel to show on startup")

    # to-notebook
    p_nb = sub.add_parser("to-notebook", help="Generate analysis notebook")
    p_nb.add_argument("file")
    p_nb.add_argument("--output", "-o", default=None,
                      help="Output .ipynb path (default: <file>.ipynb)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    try:
        import plotly_resampler  # noqa: F401
        import dash              # noqa: F401
    except ImportError:
        print(
            "ERROR: plotly-resampler and dash are required.\n"
            "Install with: pip install python-magnetrun[dashboard]",
            file=sys.stderr,
        )
        return 1

    from python_magnetrun import MagnetRun
    import python_magnetrun.dashboards as db

    if args.dashboard == "overview":
        run = MagnetRun.from_file(args.file)
        app = db.run_overview_app(run)
        app.run(port=args.port, debug=args.debug)

    elif args.dashboard == "field":
        run = MagnetRun.from_file(args.file)
        app = db.field_analysis_app(run)
        app.run(port=args.port, debug=args.debug)

    elif args.dashboard == "compare":
        if args.run_ids:
            from python_magnetrun.api import MagnetAPIClient
            client = MagnetAPIClient()
            runs = [(f"Run {rid}", client.get_run_data(rid)) for rid in args.run_ids]
        elif args.files:
            runs = [(f, MagnetRun.from_file(f)) for f in args.files]
        else:
            print("ERROR: provide file paths or --run-ids", file=sys.stderr)
            return 1
        app = db.comparison_app(runs)
        app.run(port=args.port, debug=args.debug)

    elif args.dashboard == "hybrid":
        from python_magnetrun.hybrid.hybrid_data import HybridData
        hybrid = HybridData(args.directory)
        app = db.hybrid_monitor_app(hybrid)
        app.run(port=args.port, debug=args.debug)

    elif args.dashboard == "to-notebook":
        import os
        from python_magnetrun.dashboards.notebook_generator import (
            generate_analysis_notebook,
        )
        out = args.output or os.path.splitext(args.file)[0] + "_analysis.ipynb"
        generate_analysis_notebook(args.file, out)
        print(f"Notebook written to: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add the entry point to `pyproject.toml`:

```toml
[project.scripts]
magnetrun-dashboard = "python_magnetrun.dashboards.cli:main"
```

---

## Task 4.9 — Jupyter notebook auto-generator

**File:** `python_magnetrun/dashboards/notebook_generator.py`

```python
"""
Generate pre-filled Jupyter notebooks for magnetrun analysis.

Usage::

    magnetrun-to-notebook run_20240315.txt --output analysis.ipynb
"""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


def generate_analysis_notebook(
    data_file: str,
    output: str,
    site: str = "M9",
) -> None:
    """
    Generate a pre-filled Jupyter notebook for a run data file.

    Parameters
    ----------
    data_file : str
        Path to the run data file (.txt, .tdms, .csv).
    output : str
        Output .ipynb file path.
    site : str
        Measurement site for configuration (default "M9").
    """
    cells = [
        new_markdown_cell(f"# Magnet Run Analysis\n\nData file: `{data_file}`"),

        new_code_cell(
            "# Auto-generated by magnetrun-to-notebook\n"
            "from python_magnetrun import MagnetRun\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "%matplotlib inline\n"
        ),

        new_markdown_cell("## Load data"),
        new_code_cell(
            f"run = MagnetRun.from_file({data_file!r})\n"
            f"df = run.getData()\n"
            f"print(f'Loaded {{len(df)}} rows, {{len(df.columns)}} columns')\n"
            f"df.head()"
        ),

        new_markdown_cell("## Available channels"),
        new_code_cell(
            "keys = run.getKeys()\n"
            "print(f'Channels ({len(keys)}):')\n"
            "for k in keys:\n"
            "    print(f'  {k}')"
        ),

        new_markdown_cell("## Time-series overview"),
        new_code_cell(
            "fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)\n"
            "\n"
            "t = df.get('t', df.index)\n"
            "\n"
            "# Field\n"
            "if 'Field' in df:\n"
            "    axes[0].plot(t, df['Field'], label='Field (T)')\n"
            "    axes[0].set_ylabel('Field (T)')\n"
            "    axes[0].legend()\n"
            "\n"
            "# Currents\n"
            "for key in ['IH', 'IB']:\n"
            "    if key in df:\n"
            "        axes[1].plot(t, df[key], label=f'{key} (A)')\n"
            "axes[1].set_ylabel('Current (A)')\n"
            "axes[1].legend()\n"
            "\n"
            "# Flow\n"
            "for key in ['FlowH', 'FlowB']:\n"
            "    if key in df:\n"
            "        axes[2].plot(t, df[key], label=f'{key} (l/s)')\n"
            "axes[2].set_ylabel('Flow (l/s)')\n"
            "axes[2].set_xlabel('Time (s)')\n"
            "axes[2].legend()\n"
            "\n"
            "plt.tight_layout()"
        ),

        new_markdown_cell("## Statistics"),
        new_code_cell(
            "from python_magnetrun.processing.registry import run_stat_plugins\n"
            "\n"
            "stats = {}\n"
            "for key in ['Field', 'IH', 'IB', 'FlowH', 'FlowB']:\n"
            "    if key in df.columns:\n"
            "        col = df[[key]]\n"
            "        stats[key] = {\n"
            "            'mean': col[key].mean(),\n"
            "            'min':  col[key].min(),\n"
            "            'max':  col[key].max(),\n"
            "            'std':  col[key].std(),\n"
            "        }\n"
            "\n"
            "pd.DataFrame(stats).T"
        ),

        new_markdown_cell("## Plateau detection"),
        new_code_cell(
            "from python_magnetrun.processing.registry import get_detector\n"
            "\n"
            "if 'Field' in df.columns:\n"
            "    try:\n"
            "        detect = get_detector('plateau')\n"
            "        plateaux = detect(df['Field'].values)\n"
            "        print(f'Found {len(plateaux)} plateaux')\n"
            "    except KeyError:\n"
            "        from python_magnetrun.processing.plateaux import detect_plateau\n"
            "        plateaux = detect_plateau(df['Field'].values)\n"
            "        print(f'Found {len(plateaux)} plateaux')\n"
        ),

        new_markdown_cell("## Save processed data"),
        new_code_cell(
            "import os\n"
            "out = os.path.splitext({data_file!r})[0] + '_processed.parquet'\n"
            "run.saveData(out, fmt='parquet')\n"
            "print(f'Saved to {{out}}')"
        ),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    with open(output, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Notebook written to: {output}")
```

**CLI entry point:**

```python
# Add to dashboards/cli.py inside _build_parser():
p_nb = sub.add_parser("to-notebook", help="Generate analysis notebook")
p_nb.add_argument("file", help="Run data file")
p_nb.add_argument("--output", "-o", default=None,
                  help="Output .ipynb path (default: <file>.ipynb)")
p_nb.add_argument("--site", default="M9")

# Add handler inside main():
elif args.dashboard == "to-notebook":
    from python_magnetrun.dashboards.notebook_generator import generate_analysis_notebook
    import os
    out = args.output or os.path.splitext(args.file)[0] + "_analysis.ipynb"
    generate_analysis_notebook(args.file, out, site=args.site)
    return 0
```

Add to `pyproject.toml`:

```toml
[project.scripts]
magnetrun-to-notebook = "python_magnetrun.dashboards.cli:main_notebook"
```

Or as a subcommand of `magnetrun-dashboard`:

```bash
magnetrun-dashboard to-notebook run_20240315.txt --output analysis.ipynb
```

---

## Verification Checklist

```bash
# 1. Graceful ImportError when dashboard deps not installed
pip uninstall plotly-resampler dash -y 2>/dev/null
python -c "
from python_magnetrun import dashboards
try:
    dashboards.run_overview_app(None)
except ImportError as e:
    print('Graceful ImportError:', e)
"
pip install plotly-resampler dash  # reinstall

# 2. run_overview_app builds (Dash app object returned without starting)
python -c "
import glob
from python_magnetrun import MagnetRun
from python_magnetrun.dashboards import run_overview_app
for f in glob.glob('data/*.txt')[:1]:
    run = MagnetRun.fromtxt(f)
    app = run_overview_app(run)
    print('run_overview_app OK:', type(app))
"

# 3. run_overview_widget returns FigureWidgetResampler
python -c "
import glob
from python_magnetrun import MagnetRun
from python_magnetrun.dashboards import run_overview_widget
from plotly_resampler import FigureWidgetResampler
for f in glob.glob('data/*.txt')[:1]:
    run = MagnetRun.fromtxt(f)
    fig = run_overview_widget(run)
    assert isinstance(fig, FigureWidgetResampler)
    print('widget OK, traces:', len(fig.data))
"

# 4. hybrid_monitor_app loads full-resolution data
python -c "
import os, glob
from python_magnetrun.hybrid.hybrid_data import HybridData
from python_magnetrun.dashboards import hybrid_monitor_app
dirs = glob.glob('data/hybrid*')
if dirs:
    h = HybridData(dirs[0])
    app = hybrid_monitor_app(h)
    print('hybrid_monitor_app OK:', type(app))
else:
    print('SKIP: no hybrid data directory found')
"

# 5. CLI help works
magnetrun-dashboard --help
magnetrun-dashboard overview --help
magnetrun-dashboard hybrid --help

# 6. Notebook generation
python -c "
import glob, os
from python_magnetrun.dashboards.notebook_generator import generate_analysis_notebook
for f in glob.glob('data/*.txt')[:1]:
    generate_analysis_notebook(f, '/tmp/test_analysis.ipynb')
    print('Notebook written:', os.path.exists('/tmp/test_analysis.ipynb'))
"

# 7. All tests pass
pytest tests/ -v
```

---

## Commit Strategy

```
feat(dashboards): add dashboards/ subpackage skeleton and __init__.py
feat(dashboards/figures): shared Plotly figure builders (time-series, scatter)
feat(dashboards/run_overview): FigureResampler Dash app + FigureWidgetResampler
feat(dashboards/field_analysis): field vs current Plotly dashboard
feat(dashboards/comparison): multi-run overlay with API support
feat(dashboards/hybrid_monitor): FEPC kHz/RMS viewer — FigureResampler for 10 M+ pts
feat(dashboards/cli): magnetrun-dashboard Dash CLI with overview/compare/hybrid/to-notebook
feat(dashboards/notebook): Jupyter notebook auto-generator (FigureWidgetResampler)
chore: add plotly/dash/plotly-resampler/tsdownsample to [dashboard] optional deps
chore: deprecate panels/ scripts with docstring notices
```
