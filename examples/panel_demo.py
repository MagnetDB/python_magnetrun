#!/usr/bin/env python3
"""
Panel demonstrator — MagnetRun time-series viewer.

Serves a local Panel web app in your default browser.  The reactive model
(``pn.bind``) means widgets update the chart without any page reload.

Requirements
------------
    pip install python_magnetrun[panel]

Usage
-----
    python panel_demo.py                          # argparse + pn.serve, opens browser
    python panel_demo.py run.txt --housing M9
    python panel_demo.py run.txt --keys Field Courant_A1 --port 5007

    # Panel dev-server with hot-reload (uses defaults, ignores CLI args):
    panel serve examples/panel_demo.py --show --autoreload

WASM note
---------
    panel convert examples/panel_demo.py --to pyodide-worker --out site/

    This only works if *all* imports are available inside Pyodide.
    ``python_magnetrun`` and several of its heavy dependencies (nlopt,
    nptdms, …) are not yet Pyodide-compatible, so WASM conversion is a
    future goal rather than a working feature today.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import panel as pn
import plotly.graph_objects as go

from python_magnetrun.MagnetRun import MagnetRun, load_mrun
from python_magnetrun.utils.timezone import series_utc_to_local_naive

pn.extension("plotly")

_TZ = "Europe/Paris"
_TIME_COLS = {"t", "timestamp"}

_DEFAULT_FILE = (
    Path(__file__).parent.parent / "data" / "M9_2019.02.14---23_00_38.txt"
)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(mrun: MagnetRun, keys: list[str], time_col: str) -> go.Figure:
    """Build a Plotly time-series figure from *mrun*.

    Parameters
    ----------
    mrun : MagnetRun
        Loaded run instance.
    keys : list[str]
        Channel names to plot.
    time_col : str
        ``"t"`` (elapsed seconds) or ``"timestamp"`` (UTC → Europe/Paris).

    Returns
    -------
    go.Figure
        Plotly figure with one trace per channel.
    """
    mdata = mrun.getMData()
    df = mdata.getData()

    if time_col == "timestamp" and "timestamp" in df.columns:
        t = series_utc_to_local_naive(df["timestamp"], _TZ)
        xaxis: dict = dict(
            title=f"timestamp ({_TZ})",
            type="date",
            tickformat="%d/%m/%Y<br>%H:%M:%S",
        )
    else:
        t = df.get("t", df.iloc[:, 0])
        xaxis = dict(title="t [s]")

    fig = go.Figure()
    for key in keys:
        if key in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df[key], mode="lines", name=key))

    fig.update_layout(
        title=f"{mrun.getHousing()} — {Path(mdata.FileName).name}",
        xaxis=xaxis,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def make_app(
    mrun: MagnetRun,
    initial_keys: list[str] | None = None,
    time_col: str | None = None,
) -> pn.template.BootstrapTemplate:
    """Create and wire up the Panel application.

    Parameters
    ----------
    mrun : MagnetRun
        Pre-loaded run instance (shared across sessions).
    initial_keys : list[str], optional
        Channels selected on startup; ``None`` selects all.
    time_col : str, optional
        Initial x-axis column; ``None`` auto-detects (prefers ``timestamp``).

    Returns
    -------
    pn.template.BootstrapTemplate
        Configured template — call ``.servable()`` or pass to ``pn.serve()``.
    """
    mdata = mrun.getMData()
    df = mdata.getData()
    all_keys = [k for k in mdata.getKeys() if k not in _TIME_COLS]
    avail_time = [c for c in ("timestamp", "t") if c in df.columns]
    _time_col = time_col if time_col in avail_time else (avail_time[0] if avail_time else "t")
    _keys = [k for k in initial_keys if k in all_keys] if initial_keys else all_keys

    # ── Widgets ───────────────────────────────────────────────────────────────
    key_checklist = pn.widgets.CheckBoxGroup(
        name="",
        value=_keys,
        options=all_keys,
    )
    time_radio = pn.widgets.RadioBoxGroup(
        name="",
        options=avail_time,
        value=_time_col,
        inline=False,
    )
    select_all_btn = pn.widgets.Button(
        name="Select all", button_type="default", width=100, height=30,
    )
    clear_btn = pn.widgets.Button(
        name="Clear", button_type="default", width=70, height=30,
    )

    select_all_btn.on_click(lambda _e: setattr(key_checklist, "value", all_keys))
    clear_btn.on_click(lambda _e: setattr(key_checklist, "value", []))

    # ── Reactive panes via pn.bind ────────────────────────────────────────────
    def _chart(keys: list[str], tc: str) -> pn.pane.Plotly:
        fig = build_figure(mrun, keys or [], tc)
        return pn.pane.Plotly(fig, sizing_mode="stretch_both", min_height=520)

    def _stats(keys: list[str]) -> pn.viewable.Viewable:
        if not keys:
            return pn.pane.Markdown("_No channels selected._", margin=10)
        avail = [k for k in keys if k in df.columns]
        desc = df[avail].describe().round(4)
        return pn.pane.DataFrame(
            desc, sizing_mode="stretch_width", max_rows=10, margin=10,
        )

    chart_pane = pn.bind(_chart, key_checklist, time_radio)
    stats_pane = pn.bind(_stats, key_checklist)

    # ── Layout ────────────────────────────────────────────────────────────────
    sidebar = pn.Column(
        pn.pane.Markdown("**Channels**", margin=(5, 0, 2, 0)),
        pn.Row(select_all_btn, clear_btn),
        key_checklist,
        pn.layout.Divider(),
        pn.pane.Markdown("**X-axis**", margin=(5, 0, 2, 0)),
        time_radio,
        width=210,
    )

    tabs = pn.Tabs(
        ("Chart", chart_pane),
        ("Stats", stats_pane),
        sizing_mode="stretch_both",
        dynamic=True,
    )

    return pn.template.BootstrapTemplate(
        title=f"MagnetRun — {mrun.getHousing()} — {Path(mdata.FileName).name}",
        sidebar=[sidebar],
        main=[tabs],
        sidebar_width=230,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="MagnetRun Panel viewer — interactive time-series in a browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python panel_demo.py\n"
            "  python panel_demo.py run.txt --housing M9\n"
            "  python panel_demo.py run.txt --housing M9 --keys Field Courant_A1\n"
        ),
    )
    parser.add_argument(
        "file", nargs="?", default=str(_DEFAULT_FILE),
        help="data file (.txt / .tdms / .csv)  [default: bundled M9 sample]",
    )
    parser.add_argument(
        "--housing", default="M9", metavar="NAME",
        help="housing name  [default: %(default)s]",
    )
    parser.add_argument(
        "--keys", nargs="+", metavar="KEY",
        help="channels to display initially (default: all)",
    )
    parser.add_argument(
        "--time-col", choices=["t", "timestamp"], default=None,
        help="initial x-axis column (default: auto-detect, prefers timestamp)",
    )
    parser.add_argument(
        "--port", type=int, default=5006,
        help="Panel server port  [default: %(default)s]",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Panel server host  [default: %(default)s]",
    )
    args = parser.parse_args()

    mrun = load_mrun(args.file, housing=args.housing, auto_resolve=False)
    app = make_app(mrun, initial_keys=args.keys, time_col=args.time_col)
    pn.serve(app, address=args.host, port=args.port, show=True)


# When loaded by `panel serve` (not __main__), build with defaults and
# register via .servable() so Panel can discover the app.
if __name__ == "__main__":
    main()
else:
    _mrun = load_mrun(str(_DEFAULT_FILE), housing="M9", auto_resolve=False)
    make_app(_mrun).servable()
