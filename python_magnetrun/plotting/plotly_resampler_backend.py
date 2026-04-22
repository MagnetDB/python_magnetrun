"""Plotly-resampler backend — live-kernel only.

Wraps a Plotly figure in ``FigureResampler`` (for Dash / ``show_dash()``)
or ``FigureWidgetResampler`` (for Jupyter / marimo ipywidget), enabling
view-dependent on-the-fly aggregation using ``tsdownsample`` (MinMaxLTTB).

Tier-3 downsampling strategy
-----------------------------
When this backend is selected, the ``DownsampleConfig`` pre-computation
step in ``plot_subplots()`` / ``plot_overlay()`` is skipped automatically
— ``FigureResampler`` handles downsampling dynamically on each pan/zoom.

Requires a live Python kernel — use ``PlotlyBackend`` for static export or
the REST/JS path.

Install: ``pip install python_magnetrun[resampler]``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .style import PlotStyle

logger = logging.getLogger(__name__)

__all__ = ["PlotlyResamplerBackend"]

try:
    from plotly_resampler import FigureResampler, FigureWidgetResampler

    HAS_RESAMPLER = True
except ImportError:
    HAS_RESAMPLER = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _require_resampler() -> None:
    if not HAS_RESAMPLER:
        raise ImportError(
            "plotly-resampler is required for PlotlyResamplerBackend. "
            "Install it with: pip install python_magnetrun[resampler]"
        )
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for PlotlyResamplerBackend. "
            "Install it with: pip install python_magnetrun[plotting]"
        )


class PlotlyResamplerBackend:
    """Plotly backend with dynamic, view-dependent resampling (live kernel only).

    Parameters
    ----------
    widget:
        ``True`` → ``FigureWidgetResampler`` (Jupyter / marimo ipywidget).
        ``False`` → ``FigureResampler`` (Dash / ``show_dash()``).
    """

    def __init__(self, *, widget: bool = False) -> None:
        self._widget = widget

    def subplots(
        self,
        n: int,
        *,
        share_x: bool = True,
        style: PlotStyle | None = None,
    ) -> Any:
        _require_resampler()
        s = style or PlotStyle()
        width, height_per = s.figsize
        base = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=share_x,
            vertical_spacing=0.04,
        )
        base.update_layout(
            width=width * 80,
            height=height_per * n * 80,
            template="plotly_white",
        )
        cls = FigureWidgetResampler if self._widget else FigureResampler
        return cls(base)

    def add_series(
        self,
        fig: Any,
        ax_idx: int,
        t: np.ndarray,
        y: np.ndarray,
        label: str,
        *,
        normalize: bool = False,
        color: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        _require_resampler()
        if normalize:
            abs_max = float(np.abs(y).max()) if len(y) else 1.0
            if abs_max == 0:
                abs_max = 1.0
            y = y / abs_max
            label = f"{label}  (max={abs_max:.3g})"
        scatter_kwargs: dict[str, Any] = {
            "x": t,
            "y": y,
            "name": label,
            "mode": "lines",
        }
        if color is not None:
            scatter_kwargs["line"] = {"color": color}
        fig.add_trace(go.Scatter(**scatter_kwargs), row=ax_idx + 1, col=1)
        if ylabel is not None:
            fig.update_yaxes(title_text=ylabel, row=ax_idx + 1, col=1)

    def add_annotation(
        self,
        fig: Any,
        ax_idx: int,
        t: float,
        f: float,
        label: str,
        detail: dict | None = None,
    ) -> None:
        _require_resampler()
        # Build hover text from simple scalar/string detail entries only.
        if detail:
            parts = [
                f"{k}: {v}"
                for k, v in detail.items()
                if isinstance(v, str | int | float)
            ]
            hover = "<br>".join(parts) if parts else label
        else:
            hover = label
        # Marker at (t, f)
        fig.add_trace(
            go.Scatter(
                x=[t],
                y=[f],
                mode="markers",
                marker=dict(color="yellow", size=10, symbol="circle",
                            line=dict(color="black", width=1)),
                name=label,
                showlegend=False,
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
            ),
            row=ax_idx + 1,
            col=1,
        )
        # Text annotation with arrow
        fig.add_annotation(
            x=t,
            y=f,
            text=label,
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-30,
            bgcolor="rgba(255,255,0,0.7)",
            bordercolor="gray",
            borderwidth=1,
            row=ax_idx + 1,
            col=1,
        )

    def save(self, fig: Any, path: Path, *, dpi: int = 300) -> None:
        raise NotImplementedError(
            "PlotlyResamplerBackend requires a live kernel — static export "
            "is not supported.  Use get_backend('plotly') instead."
        )

    def show(self, fig: Any) -> None:
        _require_resampler()
        if self._widget:
            # In Jupyter/marimo display() is sufficient.
            try:
                from IPython.display import display

                display(fig)
            except ImportError:
                fig.show()
        else:
            fig.show_dash()

    def to_json(self, fig: Any) -> str:
        raise NotImplementedError(
            "PlotlyResamplerBackend requires a live kernel — JSON export "
            "is not supported.  Use get_backend('plotly') instead."
        )
