#!/usr/bin/env python3
"""
Dash demonstrator — MagnetRun time-series viewer in the default web browser.

Opens a local Dash web server and displays an interactive Plotly chart.
All Plotly interactions (zoom, pan, hover, legend toggle) work natively.
Toggle channels and switch the x-axis without reloading the page.

Requirements
------------
    pip install python_magnetrun[dash]

Usage
-----
    python dash_demo.py                                          # bundled M9 sample
    python dash_demo.py data/M9_2019.02.14---23_00_38.txt --housing M9
    python dash_demo.py my_run.tdms --housing M10
    python dash_demo.py my_run.txt --housing M9 --keys Field Courant_A1
    python dash_demo.py my_run.txt --port 8051                  # custom port
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from python_magnetrun.MagnetRun import MagnetRun, load_mrun
from python_magnetrun.utils.timezone import series_utc_to_local_naive

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

    Unlike the pywry demo, Dash serialises figures via ``PlotlyJSONEncoder``
    (the same path as ``fig.to_json()``), so ``datetime64[ns]`` Series are
    converted to ISO strings automatically — no ``.dt.strftime`` workaround.

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


def _stats_table(mrun: MagnetRun, keys: list[str]) -> html.Div:
    """Return a Bootstrap table of ``describe()`` for *keys*.

    Parameters
    ----------
    mrun : MagnetRun
        Loaded run instance.
    keys : list[str]
        Selected channel names.

    Returns
    -------
    html.Div
        Div containing a responsive :class:`dbc.Table` (or a message when
        *keys* is empty).
    """
    if not keys:
        return html.Div(html.P("No channels selected.", className="text-muted p-3"))
    df = mrun.getMData().getData()
    available = [k for k in keys if k in df.columns]
    if not available:
        return html.Div(html.P("No channels available.", className="text-muted p-3"))
    desc = df[available].describe().round(4).reset_index().rename(columns={"index": "stat"})
    return html.Div(
        dbc.Table.from_dataframe(desc, striped=True, bordered=True, hover=True,
                                 responsive=True, size="sm"),
        style={"overflowX": "auto", "padding": "8px"},
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def make_app(
    mrun: MagnetRun,
    initial_keys: list[str] | None = None,
    time_col: str | None = None,
) -> dash.Dash:
    """Create and wire up the Dash application.

    Parameters
    ----------
    mrun : MagnetRun
        Pre-loaded run instance (shared across requests).
    initial_keys : list[str], optional
        Channels selected on startup; ``None`` selects all.
    time_col : str, optional
        Initial x-axis column; ``None`` auto-detects (prefers ``timestamp``).

    Returns
    -------
    dash.Dash
        Configured application — call ``.run()`` to start the server.
    """
    mdata = mrun.getMData()
    df = mdata.getData()
    all_keys = [k for k in mdata.getKeys() if k not in _TIME_COLS]

    avail_time = [c for c in ("timestamp", "t") if c in df.columns]
    _time_col = time_col if time_col in avail_time else (avail_time[0] if avail_time else "t")
    _keys = [k for k in initial_keys if k in all_keys] if initial_keys else all_keys

    # ── Layout ────────────────────────────────────────────────────────────────
    sidebar = dbc.Card(
        dbc.CardBody([
            html.H6("Channels", className="fw-bold mb-2"),
            dbc.Button(
                "Select all", id="select-all-btn", size="sm",
                color="secondary", outline=True, className="mb-2 me-1",
            ),
            dbc.Button(
                "Clear", id="clear-btn", size="sm",
                color="secondary", outline=True, className="mb-2",
            ),
            dcc.Checklist(
                id="key-checklist",
                options=[{"label": f" {k}", "value": k} for k in all_keys],
                value=_keys,
                inputStyle={"margin-right": "5px"},
                labelStyle={"display": "block", "font-size": "0.85rem"},
                className="overflow-auto",
                style={"max-height": "60vh"},
            ),
            html.Hr(),
            html.H6("X-axis", className="fw-bold mb-2"),
            dcc.RadioItems(
                id="time-col-radio",
                options=[
                    {"label": f" {c}" + (" (Europe/Paris)" if c == "timestamp" else " [s]"),
                     "value": c}
                    for c in avail_time
                ],
                value=_time_col,
                inputStyle={"margin-right": "5px"},
                labelStyle={"display": "block", "font-size": "0.85rem"},
            ),
        ]),
        className="h-100",
    )

    main_area = dbc.Tabs([
        dbc.Tab(
            dcc.Graph(id="main-graph", style={"height": "80vh"}),
            label="Chart", tab_id="tab-chart",
        ),
        dbc.Tab(
            html.Div(id="stats-div"),
            label="Stats", tab_id="tab-stats",
        ),
    ], id="main-tabs", active_tab="tab-chart")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title=f"MagnetRun — {mrun.getHousing()}",
    )

    app.layout = dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.H5(
                        f"MagnetRun — {mrun.getHousing()} — {Path(mdata.FileName).name}",
                        className="my-2 text-truncate",
                    ),
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(sidebar, width=2),
                    dbc.Col(main_area, width=10),
                ],
                style={"height": "calc(100vh - 60px)"},
            ),
        ],
        fluid=True,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("key-checklist", "value"),
        Input("select-all-btn", "n_clicks"),
        Input("clear-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_all(n_select: int | None, n_clear: int | None) -> list[str]:
        """Handle Select-all / Clear buttons."""
        triggered = dash.ctx.triggered_id
        return all_keys if triggered == "select-all-btn" else []

    @app.callback(
        Output("main-graph", "figure"),
        Output("stats-div", "children"),
        Input("key-checklist", "value"),
        Input("time-col-radio", "value"),
    )
    def update(keys: list[str] | None, tc: str) -> tuple:
        """Rebuild figure and stats whenever channel selection or x-axis changes."""
        selected = keys or []
        return build_figure(mrun, selected, tc), _stats_table(mrun, selected)

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="MagnetRun Dash viewer — interactive time-series in a browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python dash_demo.py\n"
            "  python dash_demo.py run.txt --housing M9\n"
            "  python dash_demo.py run.tdms --housing M10 --keys Field Courant_A1\n"
        ),
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=str(_DEFAULT_FILE),
        help="data file (.txt / .tdms / .csv)  [default: bundled M9 sample]",
    )
    parser.add_argument(
        "--housing", default="M9", metavar="NAME",
        help="housing name, e.g. M9, M10  [default: %(default)s]",
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
        "--port", type=int, default=8050,
        help="Dash server port  [default: %(default)s]",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Dash server host  [default: %(default)s]",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="enable Dash debug mode with hot-reload",
    )
    args = parser.parse_args()

    mrun = load_mrun(args.file, housing=args.housing, auto_resolve=False)
    app = make_app(mrun, initial_keys=args.keys, time_col=args.time_col)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
