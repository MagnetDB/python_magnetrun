"""
Textual TUI dashboard for pigbrother TDMS data files
=====================================================

Prototype demonstrator for the magnetrun TUI feature.  Displays stacked
ASCII time-series panels (one per active group) with a left sidebar of
per-group channel checkboxes.

Usage
-----
python examples/tui_dashboard.py data/M8_Overview_251105-0949.tdms
python examples/tui_dashboard.py data/M8_Archive_251105-0949.tdms

Requirements
------------
    pip install textual plotext
"""

from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Checkbox,
    Collapsible,
    Footer,
    Header,
    Label,
    Static,
    Switch,
)

# Insert repo root on path so the script works without an editable install
sys.path.insert(0, str(Path(__file__).parent.parent))
from python_magnetrun.magnetdata import load_magnetdata  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

_SKIP_GROUPS: frozenset[str] = frozenset({"Infos"})

# Groups shown with channels enabled by default (matching the reference image)
_DEFAULT_SHOWN: frozenset[str] = frozenset({"Courants_Alimentations", "Tensions_Aimant"})

_PLOT_COLORS = [
    "cyan+", "red+", "green+", "blue+", "yellow+", "magenta+", "white", "orange+",
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe_id(s: str) -> str:
    """Sanitize a TDMS name for use as a Textual widget ID."""
    return re.sub(r"[^a-zA-Z0-9]", "_", s)


def _group_timestamps(group_meta: dict) -> np.ndarray | None:
    """Return epoch-second timestamps for *group_meta*, or ``None`` for non-waveform groups.

    Parameters
    ----------
    group_meta : dict
        Channel metadata dict for one TDMS group, as stored in
        ``TdmsMagnetData.Groups[group_name]``.

    Returns
    -------
    numpy.ndarray or None
        1-D float64 array of epoch seconds, or ``None`` when the group does
        not contain waveform channels (e.g. the ``Infos`` group).
    """
    import pandas as pd

    first = next(iter(group_meta), None)
    if first is None:
        return None
    props = group_meta.get(first, {})
    if not isinstance(props, dict) or "wf_increment" not in props:
        return None

    t0 = pd.Timestamp(props["wf_start_time"])
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    t0_s = t0.timestamp() + float(props["wf_start_offset"])
    dt = float(props["wf_increment"])
    n = int(props["wf_samples"])
    return t0_s + np.arange(n, dtype=np.float64) * dt


def _render_plot(
    title: str,
    timestamps: np.ndarray,
    series: dict[str, np.ndarray],
    width: int,
    height: int,
) -> Text:
    """Render an ASCII time-series chart and return it as Rich :class:`~rich.text.Text`.

    Parameters
    ----------
    title : str
        Plot title (TDMS group name).
    timestamps : numpy.ndarray
        Epoch-second timestamps, shape ``(n,)``.
    series : dict[str, numpy.ndarray]
        Mapping of channel name → values array, shape ``(n,)``.
    width : int
        Available terminal columns.
    height : int
        Available terminal rows.

    Returns
    -------
    rich.text.Text
        ANSI-rendered plot as a Rich Text object for display in a
        :class:`textual.widgets.Static` widget.
    """
    import datetime

    import plotext as plt

    if not series or width < 12 or height < 5:
        msg = f" {title} — no channels selected" if not series else f" {title}"
        return Text(msg)

    n_out = max(8, width - 4)
    plt.clf()
    plt.plotsize(width, height)
    plt.title(title)
    plt.date_form("H:M:S")
    plt.theme("dark")

    for i, (ch, vals) in enumerate(series.items()):
        if len(vals) == 0:
            continue
        stride = max(1, len(timestamps) // n_out)
        xs = [
            datetime.datetime.utcfromtimestamp(float(t)).strftime("%H:%M:%S")
            for t in timestamps[::stride]
        ]
        ys = vals[::stride].tolist()
        plt.plot(xs, ys, label=ch, color=_PLOT_COLORS[i % len(_PLOT_COLORS)])

    return Text.from_ansi(plt.build())


# ── Widgets ──────────────────────────────────────────────────────────────────


class PlotPanel(Static):
    """ASCII time-series widget for one TDMS group.

    Parameters
    ----------
    group : str
        TDMS group name used as the plot title.
    timestamps : numpy.ndarray
        Epoch-second timestamp array for this group.
    """

    DEFAULT_CSS = """
    PlotPanel {
        height: 1fr;
        border: solid $primary-darken-2;
    }
    """

    def __init__(self, group: str, timestamps: np.ndarray, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._group = group
        self._timestamps = timestamps
        self._series: dict[str, np.ndarray] = {}

    def set_series(self, series: dict[str, np.ndarray]) -> None:
        """Store *series* data and re-render if the panel is visible.

        Parameters
        ----------
        series : dict[str, numpy.ndarray]
            Mapping of channel name → values to display.
        """
        self._series = series
        if self.display:
            self._do_render()

    def on_resize(self) -> None:
        self._do_render()

    def _do_render(self) -> None:
        w, h = self.size.width, self.size.height
        if w < 12 or h < 5:
            return
        self.update(_render_plot(self._group, self._timestamps, self._series, w, h))


# ── App ───────────────────────────────────────────────────────────────────────


class MagnetTuiApp(App[None]):
    """Textual TUI dashboard for pigbrother TDMS data.

    Parameters
    ----------
    filepath : str
        Path to a ``.tdms`` file produced by the pigbrother acquisition system.
    """

    TITLE = "Magnet TUI"
    CSS = """
    #main    { height: 1fr; }
    #sidebar {
        width: 28;
        border-right: solid $primary-darken-2;
        padding: 0 1;
        overflow-y: auto;
    }
    #plots   { width: 1fr; overflow-y: auto; }
    #enable-row { height: 3; align: left middle; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, filepath: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._filepath = filepath
        self._data = load_magnetdata(filepath)

        # Ordered list of waveform groups (Infos excluded)
        self._groups: list[str] = []
        # Epoch-second timestamp arrays keyed by group name
        self._ts: dict[str, np.ndarray] = {}
        # Enabled state: {group: {channel: bool}}
        self._state: dict[str, dict[str, bool]] = {}
        # Reverse map: checkbox widget id → (group, channel)
        self._cb_map: dict[str, tuple[str, str]] = {}

        for gname, gmeta in self._data.Groups.items():
            if gname in _SKIP_GROUPS or not isinstance(gmeta, dict):
                continue
            ts = _group_timestamps(gmeta)
            if ts is None:
                continue
            self._groups.append(gname)
            self._ts[gname] = ts
            show = gname in _DEFAULT_SHOWN
            self._state[gname] = {
                ch: show
                for ch, props in gmeta.items()
                if isinstance(props, dict) and "wf_increment" in props
            }

        for gname, channels in self._state.items():
            for ch in channels:
                cid = f"ch_{_safe_id(gname)}_{_safe_id(ch)}"
                self._cb_map[cid] = (gname, ch)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label("Pre sets")
                with Horizontal(id="enable-row"):
                    yield Label("Enable ALL  ")
                    yield Switch(id="enable-all", value=False)
                for gname in self._groups:
                    with Collapsible(
                        title=gname,
                        collapsed=gname not in _DEFAULT_SHOWN,
                        id=f"coll_{_safe_id(gname)}",
                    ):
                        for ch, enabled in self._state[gname].items():
                            yield Checkbox(
                                ch,
                                value=enabled,
                                id=f"ch_{_safe_id(gname)}_{_safe_id(ch)}",
                            )
            with Vertical(id="plots"):
                for gname in self._groups:
                    yield PlotPanel(
                        group=gname,
                        timestamps=self._ts[gname],
                        id=f"panel_{_safe_id(gname)}",
                    )
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Magnet TUI — {Path(self._filepath).name}"
        # Hide panels for groups with no enabled channels
        for gname in self._groups:
            panel = self.query_one(f"#panel_{_safe_id(gname)}", PlotPanel)
            panel.display = any(self._state[gname].values())
        # Defer series load + render until layout has settled
        self.call_after_refresh(self._initial_render)

    def _initial_render(self) -> None:
        for gname in self._groups:
            if any(self._state[gname].values()):
                self._refresh_panel(gname)

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _get_series(self, gname: str) -> dict[str, np.ndarray]:
        series: dict[str, np.ndarray] = {}
        for ch, enabled in self._state[gname].items():
            if enabled:
                df = self._data.getData(f"{gname}/{ch}")
                series[ch] = df.iloc[:, 0].to_numpy()
        return series

    def _refresh_panel(self, gname: str) -> None:
        """Show/hide and re-render the plot panel for *gname*."""
        try:
            panel = self.query_one(f"#panel_{_safe_id(gname)}", PlotPanel)
        except NoMatches:
            return
        has_enabled = any(self._state[gname].values())
        panel.display = has_enabled
        if has_enabled:
            panel.set_series(self._get_series(gname))

    # ── Event handlers ────────────────────────────────────────────────────────

    @on(Checkbox.Changed)
    def _on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        cid = event.checkbox.id
        if cid not in self._cb_map:
            return
        gname, ch = self._cb_map[cid]
        self._state[gname][ch] = bool(event.value)
        self._refresh_panel(gname)

    @on(Switch.Changed, "#enable-all")
    def _on_enable_all(self, event: Switch.Changed) -> None:
        # Update internal state first, then sync checkbox widgets
        for gname, channels in self._state.items():
            for ch in channels:
                self._state[gname][ch] = event.value
        for gname, channels in self._state.items():
            for ch in channels:
                cid = f"ch_{_safe_id(gname)}_{_safe_id(ch)}"
                with contextlib.suppress(NoMatches):
                    self.query_one(f"#{cid}", Checkbox).value = event.value
        for gname in self._groups:
            self._refresh_panel(gname)

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        """Re-render all visible plot panels (binding: r)."""
        for gname in self._groups:
            if any(self._state[gname].values()):
                self._refresh_panel(gname)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tui_dashboard",
        description="Textual TUI dashboard for pigbrother TDMS data files",
    )
    parser.add_argument("filepath", help="Path to a .tdms file")
    args = parser.parse_args()

    path = Path(args.filepath)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() != ".tdms":
        print(f"Warning: expected a .tdms file, got {path.suffix!r}", file=sys.stderr)

    try:
        MagnetTuiApp(str(path)).run()
    except ImportError as exc:
        print(
            f"Error: {exc}\n"
            "Install TUI dependencies: pip install textual plotext",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
