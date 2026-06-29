#!/usr/bin/env python3
"""
pywry demonstrator — MagnetRun time-series viewer in a native desktop window.

Opens a lightweight native webview (no browser tab required) that shows an
interactive Plotly chart.  All Plotly interactions (zoom, pan, hover, legend
click to toggle traces) work natively.

Requirements
------------
    pip install python_magnetrun[gui]          # adds pywry + plotly
    # Linux also needs: apt install python3-gi gir1.2-webkit2-4.1

Fallback
--------
If pywry is not installed the figure is saved to a temp HTML file and opened
in the default browser instead.

Usage
-----
    python pywry_demo.py                                          # bundled M9 sample
    python pywry_demo.py data/M9_2019.02.14---23_00_38.txt --housing M9
    python pywry_demo.py my_run.tdms --housing M10
    python pywry_demo.py my_run.txt --housing M9 --keys Field Courant_A1
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import webbrowser
from pathlib import Path

import plotly.graph_objects as go

from python_magnetrun.MagnetRun import load_mrun
from python_magnetrun.utils.timezone import series_utc_to_local_naive

# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(
    mrun,
    keys: list[str] | None = None,
    time_col: str | None = None,
) -> go.Figure:
    """Build a Plotly time-series figure from *mrun*.

    Parameters
    ----------
    mrun : MagnetRun
        Loaded run instance.
    keys : list[str], optional
        Channel names to include.  ``None`` includes all non-time columns.
    time_col : str, optional
        Column to use as the x-axis (``"t"`` or ``"timestamp"``).  When
        ``None`` the column is chosen automatically: ``"timestamp"`` if
        present, otherwise ``"t"``.

    Returns
    -------
    go.Figure
        Plotly figure with one trace per channel.
    """
    mdata = mrun.getMData()
    time_cols = {"t", "timestamp"}
    all_value_keys = [k for k in mdata.getKeys() if k not in time_cols]
    selected = keys if keys is not None else all_value_keys

    df = mdata.getData()
    available_time = [c for c in ("timestamp", "t") if c in df.columns]
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(
                f"time_col={time_col!r} not found in data. "
                f"Available time columns: {available_time}"
            )
        t_col = time_col
    else:
        t_col = available_time[0] if available_time else df.columns[0]

    # timestamp column is stored as naive UTC; convert to Europe/Paris for display.
    # pywry serializes the figure via _NumpyEncoder which calls .tolist() on
    # numpy.datetime64[ns] arrays — that returns int64 nanoseconds, not strings.
    # Explicitly converting to ISO strings avoids the big-int rendering bug.
    _TZ = "Europe/Paris"
    if t_col == "timestamp":
        t = series_utc_to_local_naive(df[t_col], _TZ).dt.strftime("%Y-%m-%dT%H:%M:%S")
        xaxis_label = f"timestamp ({_TZ})"
    else:
        t = df[t_col]
        xaxis_label = "t [s]"

    fig = go.Figure()
    for key in selected:
        if key not in df.columns:
            continue
        fig.add_trace(go.Scatter(x=t, y=df[key], mode="lines", name=key))

    xaxis = dict(title=xaxis_label)
    if t_col == "timestamp":
        xaxis["type"] = "date"
        xaxis["tickformat"] = "%d/%m/%Y<br>%H:%M:%S"

    fig.update_layout(
        title=f"{mrun.getHousing()} — {Path(mdata.FileName).name}",
        xaxis=xaxis,
        hovermode="x unified",
        template="plotly_white",
        height=700,
    )
    return fig


# ---------------------------------------------------------------------------
# Viewer entry point
# ---------------------------------------------------------------------------

def show(
    filepath: str | Path,
    housing: str = "unknown",
    keys: list[str] | None = None,
    time_col: str | None = None,
    *,
    width: int = 1400,
    height: int = 780,
) -> None:
    """Open *filepath* in a native pywry window (or browser fallback).

    Parameters
    ----------
    filepath : str or Path
        Path to a ``.txt``, ``.tdms``, or ``.csv`` data file.
    housing : str
        Housing identifier, e.g. ``"M9"``, ``"M10"``.
    keys : list[str], optional
        Subset of channels to display; ``None`` shows all.
    time_col : str, optional
        Column to use as the x-axis (``"t"`` or ``"timestamp"``).
        Auto-detected when ``None``.
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    """
    filepath = Path(filepath)
    mrun = load_mrun(str(filepath), housing=housing, auto_resolve=False)
    fig = build_figure(mrun, keys, time_col)
    title = f"MagnetRun — {mrun.getHousing()} — {filepath.name}"

    try:
        from pywry import Button, PyWry, Toolbar  # noqa: PLC0415
        from pywry import runtime as _pywry_runtime  # noqa: PLC0415
        handler = PyWry()

        def _on_quit(_data: object) -> None:
            # 1. Destroy all windows and cleanup (closes the webview).
            # 2. Send {"action": "quit"} to the Tauri subprocess and wait for
            #    it to exit (up to 2 s, then SIGTERM/SIGKILL).
            # 3. Force-exit Python — os._exit avoids SystemExit being swallowed
            #    by block()'s event loop in this background thread.
            handler.destroy()
            _pywry_runtime.stop()
            os._exit(0)

        exit_toolbar = Toolbar(
            position="top",
            items=[Button(label="✕ Quit", event="app:exit", variant="danger")],
        )
        handler.show_plotly(
            fig,
            title=title,
            width=width,
            height=height,
            toolbars=[exit_toolbar],
            callbacks={"app:exit": _on_quit},
        )
        handler.block()
    except ImportError:
        print(
            "pywry not installed — opening in default browser instead.\n"
            "Install with: pip install python_magnetrun[gui]",
            file=sys.stderr,
        )
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            fig.write_html(tmp.name, include_plotlyjs="cdn")
            tmp_path = tmp.name
        webbrowser.open(f"file://{tmp_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_FILE = (
    Path(__file__).parent.parent / "data" / "M9_2019.02.14---23_00_38.txt"
)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="MagnetRun pywry viewer — interactive time-series in a native window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python pywry_demo.py\n"
            "  python pywry_demo.py run.txt --housing M9\n"
            "  python pywry_demo.py run.tdms --housing M10 --keys Field Courant_A1\n"
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
        help="channels to display (default: all)",
    )
    parser.add_argument(
        "--time-col", choices=["t", "timestamp"], default=None,
        help="x-axis column: 't' (elapsed seconds) or 'timestamp' (auto-detect if omitted)",
    )
    parser.add_argument(
        "--width", type=int, default=1400,
        help="window width in pixels  [default: %(default)s]",
    )
    parser.add_argument(
        "--height", type=int, default=780,
        help="window height in pixels  [default: %(default)s]",
    )
    args = parser.parse_args()
    show(
        args.file,
        housing=args.housing,
        keys=args.keys,
        time_col=args.time_col,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
