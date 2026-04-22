"""AnnotationManager — backend-agnostic incident/event annotations.

Extracted from the procedural annotation code in ``analysis/plotting.py``
(lines 457–606).  The matplotlib implementation preserves the full
interactive pick-event / detail-subplot behaviour.  The plotly
implementation stores annotations as ``add_vline`` shapes.
"""

from __future__ import annotations

import logging
from typing import Any

from .backend import PlottingBackend
from .style import PlotColors, PlotStyle

logger = logging.getLogger(__name__)

__all__ = ["AnnotationManager"]


class AnnotationManager:
    """Manage clickable annotations on a figure.

    Parameters
    ----------
    backend:
        The active plotting backend.
    style:
        Plot style (used for detail sub-figure styling).
    colors:
        Color configuration.
    """

    def __init__(
        self,
        backend: PlottingBackend,
        style: PlotStyle | None = None,
        colors: PlotColors | None = None,
    ) -> None:
        self._backend = backend
        self._style = style or PlotStyle()
        self._colors = colors or PlotColors()

        # matplotlib-specific state
        self._mpl_annotation_dict: dict[Any, dict] = {}
        self._mpl_open_figures: dict[int, Any] = {}

    def add(
        self,
        fig: Any,
        ax_idx: int,
        t: float,
        f: float,
        label: str,
        detail: dict | None = None,
    ) -> None:
        """Add one annotation at (t, f).

        For the matplotlib backend, also stores *detail* metadata so that the
        pick handler (wired via :meth:`connect`) can open a detail sub-figure.
        For the plotly backend, delegates to ``backend.add_annotation()``.
        """
        from .matplotlib_backend import MatplotlibBackend

        if isinstance(self._backend, MatplotlibBackend):
            self._add_mpl(fig, ax_idx, t, f, label, detail)
        else:
            self._backend.add_annotation(fig, ax_idx, t, f, label, detail)

    def connect(self, fig: Any) -> None:
        """Wire up interactive events.

        For matplotlib, registers the ``pick_event`` handler that opens a
        detail sub-figure on click.  No-op for non-matplotlib backends.
        """
        from .matplotlib_backend import MatplotlibBackend

        if not isinstance(self._backend, MatplotlibBackend):
            return

        if not self._mpl_annotation_dict:
            return

        annotation_dict = self._mpl_annotation_dict
        open_figures = self._mpl_open_figures
        style = self._style
        colors = self._colors

        def on_pick(event: Any) -> None:
            if event.artist in annotation_dict:
                _show_incident_detail(
                    annotation_dict[event.artist],
                    open_figures,
                    style,
                    colors,
                )

        fig.canvas.mpl_connect("pick_event", on_pick)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_mpl(
        self,
        fig: Any,
        ax_idx: int,
        t: float,
        f: float,
        label: str,
        detail: dict | None,
    ) -> None:
        """Add an annotation at (t, f) to a matplotlib figure, and store *detail* metadata."""
        from .matplotlib_backend import MatplotlibBackend

        ax = MatplotlibBackend._get_ax(fig, ax_idx)

        (point,) = ax.plot(t, f, "yo", markersize=8)

        annot = ax.annotate(
            label,
            xy=(t, f),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc=self._colors.incident, alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )
        annot.set_picker(True)

        if detail is not None:
            self._mpl_annotation_dict[annot] = detail


def _show_incident_detail(
    metadata: dict,
    open_figures: dict[int, Any],
    style: PlotStyle,
    colors: PlotColors,
) -> None:
    """Open a detail sub-figure for a clicked incident annotation."""
    import matplotlib.pyplot as plt

    tkey: str = metadata.get("tkey", "t")
    idx: int = metadata.get("idx", 0)
    idf = metadata.get("df")
    pupitre, pupitre_key = metadata.get("pupitre", (None, None))
    archive, archive_key = metadata.get("archive", (None, None))
    anomaly: str = metadata.get("anomaly", "Incident")

    if idx in open_figures:
        plt.close(open_figures[idx])

    fig_sub, ax_sub = plt.subplots(figsize=(8, 5))

    if idf is not None and archive_key and archive_key in idf.columns:
        ax_sub.plot(
            idf[tkey].values,
            idf[archive_key].values,
            color=colors.incident,
            label=f"Incident: {archive_key}",
        )

    if pupitre is not None and not pupitre.empty and pupitre_key:
        t_start, t_end = idf[tkey].iloc[0], idf[tkey].iloc[-1]
        mask = (pupitre[tkey] >= t_start) & (pupitre[tkey] <= t_end)
        sl = pupitre[mask]
        if not sl.empty and pupitre_key in sl.columns:
            ax_sub.plot(
                sl[tkey].values,
                sl[pupitre_key].values,
                color=colors.pupitre,
                alpha=0.7,
                label=f"Pupitre: {pupitre_key}",
            )

    if archive is not None and not archive.empty and archive_key:
        t_start, t_end = idf[tkey].iloc[0], idf[tkey].iloc[-1]
        mask = (archive[tkey] >= t_start) & (archive[tkey] <= t_end)
        sl = archive[mask]
        if not sl.empty and archive_key in sl.columns:
            ax_sub.plot(
                sl[tkey].values,
                sl[archive_key].values,
                color=colors.archive,
                alpha=0.5,
                label=f"Archive: {archive_key}",
            )

    ax_sub.set_xlabel(tkey)
    if idf is not None:
        ax_sub.set_xlim(idf[tkey].iloc[0], idf[tkey].iloc[-1])
    ax_sub.set_title(anomaly)
    ax_sub.legend()
    ax_sub.grid(True, alpha=style.grid_alpha)
    fig_sub.tight_layout()
    fig_sub.show()
    open_figures[idx] = fig_sub
