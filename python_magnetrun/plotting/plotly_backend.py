"""Plotly plotting backend (static export + JSON serialisation).

Requires ``plotly`` (``pip install python_magnetrun[plotting]``).
Static image export additionally requires ``kaleido``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .style import PlotStyle

logger = logging.getLogger(__name__)

__all__ = ["PlotlyBackend"]

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _require_plotly() -> None:
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for PlotlyBackend. "
            "Install it with: pip install python_magnetrun[plotting]"
        )


class PlotlyBackend:
    """Plotting backend backed by Plotly (static + interactive)."""

    def subplots(
        self,
        n: int,
        *,
        share_x: bool = True,
        style: PlotStyle | None = None,
    ) -> Any:
        """Create a Plotly ``Figure`` with *n* sub-rows sharing the x-axis."""
        _require_plotly()
        s = style or PlotStyle()
        width, height_per = s.figsize
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=share_x,
            vertical_spacing=0.04,
        )
        fig.update_layout(
            width=width * 80,
            height=height_per * n * 80,
            template="plotly_white",
        )
        if s.grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=f"rgba(0,0,0,{s.grid_alpha})")
        else:
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
        return fig

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
        _require_plotly()
        if normalize:
            abs_max = float(np.nanmax(np.abs(y))) if len(y) else 1.0
            if abs_max == 0 or not np.isfinite(abs_max):
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
        _require_plotly()
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

    def finalize(self, fig: Any, *, xlabel: str = "t [s]") -> None:
        """Add x-axis label to the bottom row."""
        _require_plotly()
        n_rows = fig._get_subplot_rows_columns()[0][-1] if hasattr(fig, "_get_subplot_rows_columns") else 1
        fig.update_xaxes(title_text=xlabel, row=n_rows, col=1)

    def save(self, fig: Any, path: Path, *, dpi: int = 300) -> None:
        _require_plotly()
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".html", ".htm"):
            fig.write_html(str(path))
        else:
            # Requires kaleido
            try:
                fig.write_image(str(path), scale=dpi / 72)
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Plotly image export to {path} failed. "
                    "Install kaleido: pip install kaleido"
                ) from exc
        logger.info("Saved Plotly figure to %s", path)

    def show(self, fig: Any) -> None:
        _require_plotly()
        fig.show()

    def to_json(self, fig: Any) -> str:
        """Serialise the figure to a self-contained Plotly JSON string."""
        _require_plotly()
        return fig.to_json()
