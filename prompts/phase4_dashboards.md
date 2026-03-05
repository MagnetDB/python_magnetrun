# Prompt: Phase 4 — Interactive Dashboards

## Context

`python_magnetrun` currently has two non-importable scripts in `panels/`. This prompt
covers **Phase 4** of the improvement plan: replacing those scripts with a proper
`dashboards/` subpackage containing importable, composable Panel + hvplot dashboards,
a `magnetrun-dashboard` CLI, and a Jupyter notebook auto-generator.

**Prerequisite:** Phases 1, 2, 3, and 3b must be complete. `panel`, `hvplot`, and
`nbformat` must be added to the optional dependencies.

Reference document: `IMPROVEMENT_PLAN.md` §Phase 4.

---

## Objective

1. Create `python_magnetrun/dashboards/` as an importable, testable subpackage.
2. Implement `run_overview`, `field_analysis`, `comparison`, and `hybrid_monitor`
   dashboards.
3. Add a `magnetrun-dashboard` CLI that serves dashboards in a browser.
4. Add `magnetrun-to-notebook` CLI that generates pre-filled Jupyter notebooks.
5. (Optional) Wire the comparison dashboard to the `MagnetAPIClient` for multi-run
   loading from the database.

---

## Prerequisites — Add dependencies

```toml
[project.optional-dependencies]
dashboard = [
    "panel>=1.4",
    "hvplot>=0.10",
    "bokeh>=3.4",       # required by panel/hvplot
    "holoviews>=1.19",
]
notebook = [
    "nbformat>=5.10",
    "jupyter_client>=8.0",   # for notebook validation
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

## Task 4.2 — Shared widgets in `dashboards/widgets.py`

These widgets are reused across all dashboard modules.

```python
"""
Shared Panel widgets for python_magnetrun dashboards.

All widgets follow the same pattern: a factory function that returns
a configured Panel widget, and a helper to link it to a plot.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    import pandas as pd


def time_range_slider(
    df: "pd.DataFrame",
    time_col: str = "t",
    name: str = "Time range",
) -> pn.widgets.RangeSlider:
    """
    Build a time range slider bound to the data's time axis.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame with a time column.
    time_col : str
        Name of the time column (default "t").
    name : str
        Widget label.

    Returns
    -------
    pn.widgets.RangeSlider
    """
    t_min = float(df[time_col].min())
    t_max = float(df[time_col].max())
    return pn.widgets.RangeSlider(
        name=name,
        start=t_min,
        end=t_max,
        value=(t_min, t_max),
        step=(t_max - t_min) / 1000,
    )


def key_selector(
    keys: list[str],
    value: list[str] | None = None,
    name: str = "Channels",
) -> pn.widgets.CheckBoxGroup:
    """
    Multi-select checkbox for choosing which channels to display.

    Parameters
    ----------
    keys : list[str]
        All available channel names.
    value : list[str], optional
        Initially selected channels (defaults to all).
    name : str
        Widget label.

    Returns
    -------
    pn.widgets.CheckBoxGroup
    """
    return pn.widgets.CheckBoxGroup(
        name=name,
        options=keys,
        value=value if value is not None else keys,
    )


def smoothing_toggle(name: str = "Apply smoothing") -> pn.widgets.Checkbox:
    """Toggle for enabling/disabling signal smoothing."""
    return pn.widgets.Checkbox(name=name, value=False)


def smoother_selector(
    available: list[str] | None = None,
) -> pn.widgets.Select:
    """Drop-down selector for the smoothing algorithm."""
    from python_magnetrun.processing.registry import _SMOOTHERS
    options = available or list(_SMOOTHERS.keys()) or ["savgol", "lowess"]
    return pn.widgets.Select(name="Smoother", options=options)


def site_selector(sites: list[str] | None = None) -> pn.widgets.Select:
    """Drop-down selector for measurement site."""
    return pn.widgets.Select(
        name="Site",
        options=sites or ["M8", "M9", "M10"],
        value="M9",
    )
```

---

## Task 4.3 — `run_overview` dashboard

**File:** `python_magnetrun/dashboards/run_overview.py`

This is the primary dashboard — the first thing a user opens after loading a run.

```python
"""
Run overview dashboard.

Shows field, currents, flow, and temperature vs time with
regime annotations from the Signature object.

Usage::

    from python_magnetrun import MagnetRun
    from python_magnetrun.dashboards.run_overview import run_overview_dashboard
    import panel as pn

    run = MagnetRun.fromtxt("run_20240315.txt")
    dashboard = run_overview_dashboard(run)
    dashboard.servable()           # in a notebook
    # or:
    pn.serve(dashboard)            # standalone server
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import panel as pn
import param

if TYPE_CHECKING:
    from python_magnetrun import MagnetRun

logger = logging.getLogger(__name__)

# Channels displayed by default in the overview
_DEFAULT_CURRENT_KEYS = ["IH", "IB"]
_DEFAULT_FLOW_KEYS    = ["FlowH", "FlowB"]
_DEFAULT_TEMP_KEYS    = ["teb", "tsb"]


def run_overview_dashboard(
    run: "MagnetRun",
    title: str | None = None,
    height: int = 300,
    width: int = 900,
) -> pn.viewable.Viewable:
    """
    Build a Panel dashboard for a single experimental run.

    Parameters
    ----------
    run : MagnetRun
        Loaded run object.
    title : str, optional
        Dashboard title. Defaults to the run filename.
    height : int
        Height of each plot in pixels.
    width : int
        Width of each plot in pixels.

    Returns
    -------
    pn.viewable.Viewable
        A Panel Column layout containing all plots and controls.

    Examples
    --------
    >>> run = MagnetRun.fromtxt("run_20240315.txt")
    >>> dashboard = run_overview_dashboard(run)
    >>> dashboard.show()   # open in browser
    """
    import hvplot.pandas  # noqa: F401  (registers hvplot accessor)

    pn.extension()

    df = run.getData()
    keys = run.getKeys()
    title = title or f"Run: {run.getMData().FileName}"
    time_col = "t" if "t" in df.columns else df.columns[0]

    # --- Widgets ---
    from python_magnetrun.dashboards.widgets import (
        time_range_slider, smoothing_toggle, smoother_selector
    )
    t_slider   = time_range_slider(df, time_col)
    smooth_tog = smoothing_toggle()
    smoother   = smoother_selector()

    # --- Reactive plots ---
    @pn.depends(t_slider, smooth_tog, smoother)
    def field_plot(t_range, apply_smooth, smoother_name):
        if "Field" not in df.columns:
            return pn.pane.Str("No 'Field' column available")
        sub = df[(df[time_col] >= t_range[0]) & (df[time_col] <= t_range[1])]
        y = sub["Field"]
        if apply_smooth:
            try:
                from python_magnetrun.processing.registry import get_smoother
                fn = get_smoother(smoother_name)
                y = fn(y)
            except (KeyError, Exception) as e:
                logger.warning("Smoother failed: %s", e)
        return sub.assign(Field=y).hvplot.line(
            x=time_col, y="Field",
            title="Magnetic Field (T)",
            height=height, width=width,
            color="navy",
        )

    @pn.depends(t_slider, smooth_tog, smoother)
    def current_plot(t_range, apply_smooth, smoother_name):
        available = [k for k in _DEFAULT_CURRENT_KEYS if k in df.columns]
        if not available:
            return pn.pane.Str("No current columns available")
        sub = df[(df[time_col] >= t_range[0]) & (df[time_col] <= t_range[1])]
        return sub.hvplot.line(
            x=time_col, y=available,
            title="Currents (A)",
            height=height, width=width,
        )

    @pn.depends(t_slider)
    def flow_plot(t_range):
        available = [k for k in _DEFAULT_FLOW_KEYS if k in df.columns]
        if not available:
            return pn.pane.Str("No flow columns available")
        sub = df[(df[time_col] >= t_range[0]) & (df[time_col] <= t_range[1])]
        return sub.hvplot.line(
            x=time_col, y=available,
            title="Flow Rates (l/s)",
            height=height, width=width,
        )

    @pn.depends(t_slider)
    def temperature_plot(t_range):
        available = [k for k in _DEFAULT_TEMP_KEYS if k in df.columns]
        if not available:
            return pn.pane.Str("No temperature columns available")
        sub = df[(df[time_col] >= t_range[0]) & (df[time_col] <= t_range[1])]
        return sub.hvplot.line(
            x=time_col, y=available,
            title="Temperatures (°C)",
            height=height, width=width,
        )

    # --- Stats table ---
    def _stats_table() -> pn.pane.DataFrame:
        rows = []
        for key in ["Field", "IH", "IB", "FlowH", "FlowB"]:
            if key in df.columns:
                rows.append({
                    "Channel": key,
                    "Mean": f"{df[key].mean():.3f}",
                    "Min":  f"{df[key].min():.3f}",
                    "Max":  f"{df[key].max():.3f}",
                    "Std":  f"{df[key].std():.3f}",
                })
        import pandas as pd
        return pn.pane.DataFrame(pd.DataFrame(rows), index=False)

    # --- Layout ---
    controls = pn.Column(
        pn.pane.Markdown(f"## {title}"),
        t_slider,
        pn.Row(smooth_tog, smoother),
    )

    plots = pn.Column(
        pn.panel(field_plot),
        pn.panel(current_plot),
        pn.panel(flow_plot),
        pn.panel(temperature_plot),
    )

    stats = pn.Column(
        pn.pane.Markdown("### Summary statistics"),
        _stats_table(),
    )

    return pn.Column(controls, plots, stats)
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

```python
"""
Hybrid FEPC data monitor dashboard.

Displays kHz, RMS, and trigger channel data from HybridRun objects.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from python_magnetrun.hybrid.hybrid_data import HybridData

logger = logging.getLogger(__name__)


def hybrid_monitor_dashboard(
    hybrid: "HybridData",
    max_points: int = 10_000,
) -> pn.viewable.Viewable:
    """
    Build a dashboard for FEPC hybrid acquisition data.

    Parameters
    ----------
    hybrid : HybridData
        Loaded hybrid data object.
    max_points : int
        Maximum number of points to display (downsampled via LTTB if exceeded).

    Returns
    -------
    pn.viewable.Viewable
    """
    import hvplot.pandas  # noqa: F401
    pn.extension()

    # Channel selectors
    kHz_keys  = getattr(hybrid, "kHz_keys",  [])
    rms_keys  = getattr(hybrid, "rms_keys",  [])
    trig_keys = getattr(hybrid, "trigger_keys", [])

    kHz_select  = pn.widgets.Select(name="kHz channel",  options=kHz_keys  or ["(none)"])
    rms_select  = pn.widgets.Select(name="RMS channel",  options=rms_keys  or ["(none)"])
    trig_select = pn.widgets.Select(name="Trigger channel", options=trig_keys or ["(none)"])

    @pn.depends(kHz_select)
    def kHz_plot(key):
        if key == "(none)":
            return pn.pane.Str("No kHz channels")
        try:
            df = hybrid.read_kHz(key, max_points=max_points)
            return df.hvplot.line(title=f"kHz: {key}", height=300, width=900)
        except Exception as exc:
            return pn.pane.Str(f"Error loading {key}: {exc}")

    @pn.depends(rms_select)
    def rms_plot(key):
        if key == "(none)":
            return pn.pane.Str("No RMS channels")
        try:
            df = hybrid.read_rms(key, max_points=max_points)
            return df.hvplot.line(title=f"RMS: {key}", height=300, width=900)
        except Exception as exc:
            return pn.pane.Str(f"Error loading {key}: {exc}")

    return pn.Column(
        pn.pane.Markdown("## Hybrid FEPC Monitor"),
        pn.Row(kHz_select, rms_select, trig_select),
        pn.panel(kHz_plot),
        pn.panel(rms_plot),
    )
```

---

## Task 4.7 — `dashboards/__init__.py`

```python
"""
python_magnetrun interactive dashboards.

All dashboards require: panel, hvplot, bokeh.
Install with: pip install python-magnetrun[dashboard]

Available dashboards
--------------------
- run_overview_dashboard(run)        — time-series overview of a single run
- field_analysis_dashboard(run)      — B vs I curves and hysteresis
- comparison_dashboard(runs)         — multi-run overlay and statistics
- comparison_dashboard_from_api(...) — load runs from the API and compare
- hybrid_monitor_dashboard(hybrid)   — FEPC kHz/RMS/trigger data viewer
"""
from __future__ import annotations


def _require_panel() -> None:
    try:
        import panel  # noqa: F401
        import hvplot  # noqa: F401
    except ImportError:
        raise ImportError(
            "Dashboards require panel and hvplot. "
            "Install with: pip install python-magnetrun[dashboard]"
        )


def run_overview_dashboard(run, **kwargs):
    _require_panel()
    from python_magnetrun.dashboards.run_overview import run_overview_dashboard as _fn
    return _fn(run, **kwargs)


def field_analysis_dashboard(run, **kwargs):
    _require_panel()
    from python_magnetrun.dashboards.field_analysis import field_analysis_dashboard as _fn
    return _fn(run, **kwargs)


def comparison_dashboard(runs, **kwargs):
    _require_panel()
    from python_magnetrun.dashboards.comparison import comparison_dashboard as _fn
    return _fn(runs, **kwargs)


def comparison_dashboard_from_api(run_ids, client=None):
    _require_panel()
    from python_magnetrun.dashboards.comparison import comparison_dashboard_from_api as _fn
    return _fn(run_ids, client=client)


def hybrid_monitor_dashboard(hybrid, **kwargs):
    _require_panel()
    from python_magnetrun.dashboards.hybrid_monitor import hybrid_monitor_dashboard as _fn
    return _fn(hybrid, **kwargs)


__all__ = [
    "run_overview_dashboard",
    "field_analysis_dashboard",
    "comparison_dashboard",
    "comparison_dashboard_from_api",
    "hybrid_monitor_dashboard",
]
```

---

## Task 4.8 — `magnetrun-dashboard` CLI

**File:** `python_magnetrun/dashboards/cli.py`

```python
"""
CLI for serving python_magnetrun dashboards.

Usage::

    magnetrun-dashboard overview run_20240315.txt
    magnetrun-dashboard compare run_A.txt run_B.txt
    magnetrun-dashboard compare --run-ids 42 43 --via-api
    magnetrun-dashboard hybrid  /path/to/hybrid_dir/
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="magnetrun-dashboard",
        description="Serve a python_magnetrun interactive dashboard",
    )
    parser.add_argument("--port", type=int, default=5006,
                        help="Port to serve on (default: 5006)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="dashboard", required=True)

    # overview
    p_ov = sub.add_parser("overview", help="Run overview dashboard")
    p_ov.add_argument("file", help="Path to the run data file (.txt, .tdms, .csv)")
    p_ov.add_argument("--site", help="Site override (M8, M9, M10)")

    # field
    p_fi = sub.add_parser("field", help="Field analysis dashboard")
    p_fi.add_argument("file")

    # compare
    p_cm = sub.add_parser("compare", help="Multi-run comparison")
    p_cm.add_argument("files", nargs="*", help="Run data files to compare")
    p_cm.add_argument("--run-ids", nargs="+", type=int, metavar="ID",
                      help="Run IDs to load from the API")
    p_cm.add_argument("--api-url", help="API base URL (overrides MAGNETAPI_URL)")

    # hybrid
    p_hy = sub.add_parser("hybrid", help="Hybrid FEPC data monitor")
    p_hy.add_argument("directory", help="Directory containing hybrid FEPC files")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    try:
        import panel as pn
    except ImportError:
        print(
            "ERROR: panel is required for dashboards.\n"
            "Install with: pip install python-magnetrun[dashboard]",
            file=sys.stderr,
        )
        return 1

    from python_magnetrun import MagnetRun
    import python_magnetrun.dashboards as db

    open_browser = not args.no_browser

    if args.dashboard == "overview":
        run = MagnetRun.from_file(args.file)
        dashboard = db.run_overview_dashboard(run)
        pn.serve(dashboard, port=args.port, show=open_browser,
                 title="Run Overview")

    elif args.dashboard == "field":
        run = MagnetRun.from_file(args.file)
        dashboard = db.field_analysis_dashboard(run)
        pn.serve(dashboard, port=args.port, show=open_browser,
                 title="Field Analysis")

    elif args.dashboard == "compare":
        if args.run_ids:
            dashboard = db.comparison_dashboard_from_api(
                args.run_ids,
                client=None,  # reads MAGNETAPI_URL from env
            )
        else:
            if not args.files:
                print("ERROR: provide file paths or --run-ids", file=sys.stderr)
                return 1
            runs = [(f, MagnetRun.from_file(f)) for f in args.files]
            dashboard = db.comparison_dashboard(runs)
        pn.serve(dashboard, port=args.port, show=open_browser,
                 title="Run Comparison")

    elif args.dashboard == "hybrid":
        from python_magnetrun.hybrid.hybrid_data import HybridData
        hybrid = HybridData(args.directory)
        dashboard = db.hybrid_monitor_dashboard(hybrid)
        pn.serve(dashboard, port=args.port, show=open_browser,
                 title="Hybrid Monitor")

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
# 1. Import without dashboard dependencies installed (graceful error)
pip uninstall panel hvplot -y 2>/dev/null
python -c "
from python_magnetrun import dashboards
try:
    dashboards.run_overview_dashboard(None)
except ImportError as e:
    print('Graceful ImportError:', e)
"
pip install panel hvplot  # reinstall

# 2. run_overview builds without error
python -c "
import glob
from python_magnetrun import MagnetRun
from python_magnetrun.dashboards import run_overview_dashboard
for f in glob.glob('data/*.txt')[:1]:
    run = MagnetRun.fromtxt(f)
    dashboard = run_overview_dashboard(run)
    print('overview OK:', type(dashboard))
"

# 3. comparison_dashboard with two local files
python -c "
import glob
from python_magnetrun import MagnetRun
from python_magnetrun.dashboards import comparison_dashboard
files = glob.glob('data/*.txt')[:2]
if len(files) >= 2:
    runs = [(f, MagnetRun.fromtxt(f)) for f in files]
    d = comparison_dashboard(runs)
    print('comparison OK:', type(d))
"

# 4. CLI help works
magnetrun-dashboard --help
magnetrun-dashboard overview --help
magnetrun-dashboard compare --help

# 5. Notebook generation
magnetrun-dashboard to-notebook data/$(ls data/*.txt | head -1 | xargs basename)
ls *.ipynb

# 6. All tests pass
pytest tests/ -v
```

---

## Commit Strategy

```
feat(dashboards): add dashboards/ subpackage skeleton and __init__.py
feat(dashboards/widgets): shared Panel widgets (time slider, key selector, smoother)
feat(dashboards/run_overview): run overview dashboard with reactive plots
feat(dashboards/field_analysis): field vs current dashboard
feat(dashboards/comparison): multi-run comparison with API support
feat(dashboards/hybrid_monitor): FEPC kHz/RMS data viewer
feat(dashboards/cli): magnetrun-dashboard CLI with overview/compare/hybrid/to-notebook
feat(dashboards/notebook): Jupyter notebook auto-generator
chore: add panel/hvplot/nbformat to optional dependencies
chore: deprecate panels/ scripts with docstring notices
```
