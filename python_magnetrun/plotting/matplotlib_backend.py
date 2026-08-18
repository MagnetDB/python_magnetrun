"""Matplotlib plotting backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .style import LabelStyle, PlotStyle

logger = logging.getLogger(__name__)

__all__ = ["MatplotlibBackend"]


def _mpl_text_kwargs(label_style: LabelStyle | None, fallback_size: int) -> dict[str, Any]:
    """Build matplotlib text keyword arguments from a :class:`LabelStyle`.

    Parameters
    ----------
    label_style : LabelStyle, optional
        Source style; ``None`` returns only ``{"fontsize": fallback_size}``.
    fallback_size : int
        Used when ``label_style`` is ``None`` or ``label_style.size`` is ``None``.

    Returns
    -------
    dict
        Keyword arguments suitable for ``set_xlabel``, ``set_ylabel``,
        ``set_title``, ``annotate``, etc.
    """
    if label_style is None:
        return {"fontsize": fallback_size}
    kwargs: dict[str, Any] = {"fontsize": label_style.size if label_style.size is not None else fallback_size}
    if label_style.family is not None:
        kwargs["fontfamily"] = label_style.family
    if label_style.style is not None:
        kwargs["fontstyle"] = label_style.style
    if label_style.weight is not None:
        kwargs["fontweight"] = label_style.weight
    if label_style.color is not None:
        kwargs["color"] = label_style.color
    return kwargs


class MatplotlibBackend:
    """Plotting backend backed by matplotlib."""

    def subplots(
        self,
        n: int,
        *,
        share_x: bool = True,
        style: PlotStyle | None = None,
    ) -> Any:
        """Create *n* stacked subplots sharing the x-axis.

        Returns the ``matplotlib.figure.Figure`` object; axes are accessible
        via ``fig.axes``.
        """
        s = style or PlotStyle()
        width, height_per = s.figsize
        fig, axes = plt.subplots(n, 1, sharex=share_x, figsize=(width, height_per * n))
        ax_list = [axes] if n == 1 else list(axes)
        fig._magnetrun_axes = ax_list
        fig._magnetrun_style = s
        if s.grid:
            for ax in ax_list:
                ax.grid(True, alpha=s.grid_alpha)
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
        marker: str | None = None,
        linestyle: str | None = None,
        markevery: int | None = None,
        alpha: float | None = None,
    ) -> None:
        ax = self._get_ax(fig, ax_idx)
        if normalize:
            abs_max = float(np.nanmax(np.abs(y))) if len(y) else 1.0
            if abs_max == 0 or not np.isfinite(abs_max):
                abs_max = 1.0
            y = y / abs_max
            label = f"{label}  (max={abs_max:.3g})"
        kwargs: dict[str, Any] = {"label": label}
        if color is not None:
            kwargs["color"] = color
        if marker is not None:
            kwargs["marker"] = marker
        if linestyle is not None:
            kwargs["linestyle"] = linestyle
        if markevery is not None:
            kwargs["markevery"] = markevery
        if alpha is not None:
            kwargs["alpha"] = alpha

        # Filter out NaN values to avoid matplotlib rendering issues with sparse arrays.
        # When plotting merged data from multiple sources, NaN gaps can cause matplotlib
        # to incorrectly render only small segments. Filtering preserves the correct
        # visual representation while allowing line breaks at NaN boundaries.
        valid_mask = ~np.isnan(y)
        if valid_mask.any():
            ax.plot(t[valid_mask], y[valid_mask], **kwargs)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

    def add_scatter(
        self,
        fig: Any,
        ax_idx: int,
        t: np.ndarray,
        y: np.ndarray,
        label: str,
        *,
        color: str = "red",
        size: int = 10,
        alpha: float = 0.7,
    ) -> None:
        ax = self._get_ax(fig, ax_idx)
        ax.scatter(t, y, c=color, s=size, alpha=alpha, label=label, zorder=5)

    def add_annotation(
        self,
        fig: Any,
        ax_idx: int,
        t: float,
        f: float,
        label: str,
        detail: dict | None = None,
    ) -> None:
        s: PlotStyle = getattr(fig, "_magnetrun_style", None) or PlotStyle()
        annot_kwargs = _mpl_text_kwargs(s.annotation_style, fallback_size=8)
        ax = self._get_ax(fig, ax_idx)
        ax.axvline(t, linestyle="--", color="gray", alpha=0.6)
        ax.annotate(
            label,
            xy=(t, f),
            xytext=(5, -15),
            textcoords="offset points",
            rotation=90,
            **annot_kwargs,
        )

    def save(self, fig: Any, path: Path, *, dpi: int = 300) -> None:
        self.finalize(fig)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {path}")

    def finalize(self, fig: Any, *, xlabel: str = "t [s]") -> None:
        """Add legend, axis labels, and text styling to all axes."""
        s: PlotStyle = getattr(fig, "_magnetrun_style", None) or PlotStyle()
        axes = getattr(fig, "_magnetrun_axes", None) or fig.axes
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()
        # plot_xy() / plot_data() store custom labels on the figure; honour them.
        _xlabel = getattr(fig, "_magnetrun_xlabel", xlabel)
        if axes and _xlabel and not axes[-1].get_xlabel():
            axes[-1].set_xlabel(_xlabel)
        _ylabel = getattr(fig, "_magnetrun_ylabel", None)
        if axes and _ylabel and not axes[0].get_ylabel():
            axes[0].set_ylabel(_ylabel)

        # Apply LabelStyle font properties to already-set text elements.
        xl_kwargs = _mpl_text_kwargs(s.xlabel_style, fallback_size=s.label_fontsize)
        yl_kwargs = _mpl_text_kwargs(s.ylabel_style, fallback_size=s.label_fontsize)
        for ax in axes:
            if ax.get_xlabel():
                ax.xaxis.label.set(**xl_kwargs)
            if ax.get_ylabel():
                ax.yaxis.label.set(**yl_kwargs)

        # Apply title style to the suptitle (fig-level text object, if present).
        if fig.texts:
            t_kwargs = _mpl_text_kwargs(s.title_style, fallback_size=s.title_fontsize)
            fig.texts[0].set(**t_kwargs)

    def show(self, fig: Any) -> None:
        self.finalize(fig)
        plt.figure(fig.number)
        plt.show()

    def to_json(self, fig: Any) -> str:
        raise NotImplementedError(
            "MatplotlibBackend does not support to_json(). "
            "Switch to get_backend('plotly') for JSON/REST output."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ax(fig: Any, ax_idx: int) -> Any:
        axes = getattr(fig, "_magnetrun_axes", None) or fig.axes
        return axes[ax_idx]
