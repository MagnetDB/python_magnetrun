"""
Plotting utilities for magnetrun analysis.

This module provides visualization tools for comparing data from
multiple sources (overview, archive, pupitre, incidents) with support
for large datasets through downsampling.

Features:
- Downsampling for efficient plotting of large datasets
- Multi-source data comparison plots
- Interactive incident annotations
- Regime visualization with colored spans
- Configurable styles and colors

Example usage::

    from python_magnetrun.analysis.plotting import (
        plot_data,
        plot_comparison,
    )

    # Plot with 10% downsampling for large datasets
    plot_data(
        df_overview, df_archive, df_pupitre, df_incidents,
        channels_dict, pupitre_dict, site,
        tkey="t", key="Courant_GR1",
        title="M9 Run", msg="(synchronized)",
        downsample_percent=10.0,
        show=True, save=False,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python_magnetrun.plotting.backend import PlottingBackend, get_backend
from python_magnetrun.plotting.style import (  # noqa: F401
    DEFAULT_COLORS,
    DEFAULT_STYLE,
    PlotColors,
    PlotStyle,
)
from python_magnetrun.plotting.timeseries import plot_overlay
from python_magnetrun.plotting.utils import format_axis_label
from python_magnetrun.utils.downsampling import DownsampleConfig, downsample_arrays  # noqa: F401

# Module logger
logger = logging.getLogger("python_magnetrun.analysis.plotting")



def estimate_downsample_percent(
    n_points: int,
    target_points: int = 10000,
) -> float:
    """
    Estimate appropriate downsample percentage for a given dataset size.

    Parameters
    ----------
    n_points : int
        Number of data points
    target_points : int, optional
        Target number of points for plotting (default: 10000)

    Returns
    -------
    float
        Recommended downsample percentage (0-100)
    """
    if n_points <= target_points:
        return 100.0

    return (target_points / n_points) * 100.0


# =============================================================================
# Main plotting functions
# =============================================================================
def plot_data(
    df_overview: pd.DataFrame,
    df_archive: pd.DataFrame,
    df_pupitre: pd.DataFrame,
    df_incidents: dict[str, list[pd.DataFrame]] | None,
    channels_dict: dict[str, str],
    pupitre_dict: dict[str, dict[str, str]],
    housing: str,
    tkey: str,
    key: str,
    title: str,
    msg: str,
    show: bool = False,
    save: bool = False,
    output_path: str | None = None,
    downsample_percent: float = 100.0,
    downsample_config: DownsampleConfig | None = None,
    style: PlotStyle | None = None,
    colors: PlotColors | None = None,
    interactive: bool = True,
    backend: str | PlottingBackend | None = None,
    df_hybrid: pd.DataFrame | None = None,
    df_hybrid_incidents: dict[str, list[pd.DataFrame]] | None = None,
    hybrid_dict: dict[str, dict[str, str]] | None = None,
) -> Any | None:
    """
    Plot data from multiple sources with optional downsampling.

    Creates a comparison plot showing overview, archive, and pupitre data
    on the same axes, with optional incident markers.

    Parameters
    ----------
    df_overview : pd.DataFrame
        Overview data (1 Hz sampling)
    df_archive : pd.DataFrame
        Archive data (120 Hz sampling)
    df_pupitre : pd.DataFrame
        Pupitre data (variable sampling)
    df_incidents : dict or None
        Dictionary mapping incident types to lists of DataFrames
        e.g., {"default": [df1, df2], "trigger": [df3], "spike": [df4]}
    channels_dict : dict
        Mapping from key names to channel names
    pupitre_dict : dict
        Housing-specific mapping for pupitre channels
    housing : str
        Housing identifier (e.g., "M9")
    tkey : str
        Time column name (e.g., "t" or "timestamp")
    key : str
        Data key to plot (e.g., "Courant_GR1")
    title : str
        Plot title
    msg : str
        Additional message for title (e.g., sync status)
    show : bool, optional
        Display plot interactively
    save : bool, optional
        Save plot to file
    output_path : str, optional
        Output file path (auto-generated if not provided)
    downsample_config : DownsampleConfig or None, optional
        Downsampling configuration (method + n_out + bucket_size).
        ``None`` disables downsampling (default).
    style : PlotStyle, optional
        Plot style configuration
    colors : PlotColors, optional
        Color configuration
    interactive : bool, optional
        Enable interactive incident annotations
    backend : str or PlottingBackend, optional
        Backend name or instance (default: auto-detected).
    df_hybrid : pandas.DataFrame, optional
        Hybrid data to overlay on the same axes; ``None`` skips.
    df_hybrid_incidents : dict, optional
        Incident DataFrames for hybrid data, same structure as
        *df_incidents*; ``None`` skips.
    hybrid_dict : dict, optional
        Housing-specific mapping for hybrid channels, same structure as
        *pupitre_dict*; ``None`` skips.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, None otherwise

    Examples
    --------
    >>> plot_data(
    ...     df_overview, df_archive, df_pupitre, df_incidents,
    ...     channels_dict, pupitre_dict, "M9",
    ...     tkey="t", key="Courant_GR1",
    ...     title="M9_Overview_241106", msg="(synchronized)",
    ...     downsample_config=DownsampleConfig(n_out=10000, method="minmax_lttb"),
    ...     show=True,
    ... )
    """

    logger.info(
        f"Plotting data for key '{key}' vs tkey={tkey} with downsample_config={downsample_config!r}"
    )

    style = style or DEFAULT_STYLE
    colors = colors or DEFAULT_COLORS
    b = get_backend(backend)

    # ------------------------------------------------------------------
    # Build one merged DataFrame: one row per sample, NaN where a source
    # has no value at that timestamp.  plot_overlay handles NaN as gaps.
    # ------------------------------------------------------------------
    dfs_to_concat: list[pd.DataFrame] = []
    field_names: list[str] = []
    field_colors: list[str] = []
    field_styles: list[tuple] = []

    if not df_overview.empty:
        logger.debug(
            f"housing={housing}, key={key}, channels_dict={channels_dict}, "
            f"df.columns={df_overview.columns.tolist()}"
        )
        ov_rename: dict[str, str] = {}
        if key in df_overview.columns:
            col = f"Overview: {key}"
            ov_rename[key] = col
            field_names.append(col)
            field_colors.append(colors.overview)
            field_styles.append((None, None, None, 1))
        ov_ch = channels_dict.get(key)
        if ov_ch and ov_ch in df_overview.columns:
            col = f"Overview: {ov_ch}"
            ov_rename[ov_ch] = col
            field_names.append(col)
            field_colors.append(colors.overview)
            field_styles.append(
                (None, ".", None, 1)
            )  # small markers distinguish source
        if ov_rename:
            ov_cols = [tkey] + list(ov_rename.keys())
            dfs_to_concat.append(df_overview[ov_cols].rename(columns=ov_rename))

    if not df_archive.empty:
        logger.debug(
            f"housing={housing}, key={key}, channels_dict={channels_dict}, "
            f"df.columns={df_archive.columns.tolist()}"
        )
        ar_ch = channels_dict.get(key)
        if ar_ch and ar_ch in df_archive.columns:
            col = f"Archive: {ar_ch}"
            dfs_to_concat.append(df_archive[[tkey, ar_ch]].rename(columns={ar_ch: col}))
            field_names.append(col)
            field_colors.append(colors.archive)
            field_styles.append(
                ("--", None, None, 0.5)
            )  # dashed + semi-transparent to distinguish

    if not df_pupitre.empty:
        logger.debug(
            f"housing={housing}, key={key}, pupitre_dict={pupitre_dict}, "
            f"df.columns={df_pupitre.columns.tolist()}"
        )
        pu_ch = pupitre_dict.get(housing, {}).get(key)
        if pu_ch and pu_ch in df_pupitre.columns:
            col = f"Pupitre: {pu_ch}"
            dfs_to_concat.append(df_pupitre[[tkey, pu_ch]].rename(columns={pu_ch: col}))
            field_names.append(col)
            field_colors.append(colors.pupitre)
            field_styles.append(("-.", None, None, 1))  # dash-dot for pupitre
            logger.debug(f"pupitre[{pu_ch}]:\n{dfs_to_concat[-1].head()}")
            logger.debug(f"pupitre[{pu_ch}]:\n{dfs_to_concat[-1].describe()}")

    if df_hybrid is not None and not df_hybrid.empty:
        hy_ch = (hybrid_dict or {}).get(housing, {}).get(key)
        if hy_ch and hy_ch in df_hybrid.columns:
            col = f"Hybrid: {hy_ch}"
            dfs_to_concat.append(df_hybrid[[tkey, hy_ch]].rename(columns={hy_ch: col}))
            field_names.append(col)
            field_colors.append(colors.hybrid)
            field_styles.append(
                (":", None, None, 1)
            )  # dotted to distinguish from pupitre

    if not field_names:
        logger.warning(
            f"plot_data: no data columns found for key {key!r} — nothing to plot"
        )
        return None

    # Resolve unit for ylabel before concat drops attrs
    _ch = channels_dict.get(key, key)
    logger.debug(
        f"Resolving unit for key={key!r}, channel={_ch!r} from overview and archive metadata"
    )
    for _df in [
        df_overview,
        df_archive,
        df_pupitre,
    ]:
        logger.debug(f"{_df}.attrs.get('units', {{}}): {_df.attrs.get('units', {})}")

    _unit_entry = (
        df_overview.attrs.get("units", {}).get(_ch)
        or df_overview.attrs.get("units", {}).get(key)
        or df_archive.attrs.get("units", {}).get(_ch)
        or df_archive.attrs.get("units", {}).get(key)
    )
    logger.debug(
        f"Found unit entry for key={key!r}: {_unit_entry!r} (type: {type(_unit_entry)})"
    )
    _pint_unit = _unit_entry[1] if _unit_entry else None
    _symbol_unit = _unit_entry[0] if _unit_entry else ""
    ylabel = format_axis_label(_symbol_unit, _pint_unit)
    logger.debug(
        f"Determined ylabel={ylabel!r} for key={key!r} with unit entry: {_unit_entry!r}"
    )

    merged = (
        pd.concat(dfs_to_concat, ignore_index=True)
        .sort_values(tkey, kind="stable")
        .reset_index(drop=True)
    )
    logger.debug(
        f"Merged DataFrame for plotting:\n{merged.head()}\n{merged.describe()}"
    )

    from python_magnetrun.plotting.matplotlib_backend import MatplotlibBackend

    # For matplotlib, call plt.subplots directly so the figure is created via
    # the module-level plt import (required for mock-based testing).
    _direct_mpl = isinstance(b, MatplotlibBackend)
    if _direct_mpl:
        logger.debug(
            f"Using backend {type(b).__name__} to plot data with fields: {field_names} -- in direct matplotlib mode"
        )
        t_arr = merged[tkey].to_numpy(dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=style.figsize)
        fig._magnetrun_axes = [ax]  # type: ignore[attr-defined]
        for i, field in enumerate(field_names):
            logger.debug(
                f"Plotting field '{field}' with color {field_colors[i]} and style {field_styles[i]}"
            )
            if field not in merged.columns:
                continue
            y_arr = merged[field].to_numpy(dtype=float)
            # Filter to valid (non-NaN) points per field. Without this, sparse
            # sources (e.g. 1 Hz overview mixed with 120 Hz archive) produce
            # isolated single points surrounded by NaN gaps; matplotlib renders
            # zero-length line segments which are invisible with no marker.
            valid = ~np.isnan(y_arr)
            if not valid.any():
                continue
            color = field_colors[i] if i < len(field_colors) else None
            ls, mk, me, al = (
                field_styles[i] if i < len(field_styles) else (None, None, None, None)
            )
            _plot_kwargs: dict = {
                "label": field,
                "color": color,
                "linestyle": ls or "-",
            }
            if mk:
                _plot_kwargs["marker"] = mk
            if me is not None:
                _plot_kwargs["markevery"] = me
            if al is not None:
                _plot_kwargs["alpha"] = al
            ax.plot(t_arr[valid], y_arr[valid], **_plot_kwargs)
        ax.legend()
        ax.set_xlabel(f"{tkey} [s]")
        ax.set_ylabel(ylabel)
        if style.grid:
            ax.grid(True, alpha=style.grid_alpha)
        plot_title = f"{title.replace('_Overview', '')}: {key} {msg}"
        if plot_title:
            fig.suptitle(plot_title)
    else:
        logger.debug(
            f"Using backend {type(b).__name__} to plot data with fields: {field_names}"
        )
        fig = plot_overlay(
            merged,
            field_names,
            t_col=tkey,
            backend=b,
            style=style,
            title=f"{title.replace('_Overview', '')}: {key} {msg}",
            colors=field_colors,
            field_styles=field_styles,
            downsample=downsample_config,
        )
        fig._magnetrun_xlabel = f"{tkey} [s]"  # type: ignore[attr-defined]
        fig._magnetrun_ylabel = ylabel  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    _has_incidents = df_incidents is not None and any(
        len(lst) > 0 for lst in df_incidents.values()
    )
    logger.debug(
        f"_has_incidents={_has_incidents} based on df_incidents: {df_incidents}"
    )
    logger.debug(
        f"df_hybrid_incidents: {df_hybrid_incidents if df_hybrid_incidents is not None else 'None'}"
    )
    _has_hybrid_incidents = (
        df_hybrid_incidents is not None
        and len(df_hybrid_incidents) > 0
        and any(len(lst) > 0 for lst in df_hybrid_incidents.values())
    )
    if interactive and (_has_incidents or _has_hybrid_incidents):
        from python_magnetrun.plotting.annotations import AnnotationManager

        manager = AnnotationManager(b, style=style, colors=colors)

        if df_incidents is not None:
            logger.info(f"Adding incident annotations: {list(df_incidents.keys())}")
            for itype, incident_list in df_incidents.items():
                for i, idf in enumerate(incident_list):
                    if idf.empty:
                        continue
                    t_mid = idf[tkey].median()
                    incident_key = channels_dict.get(key, key)
                    if incident_key not in idf.columns:
                        continue
                    f_mid = idf[incident_key].median()
                    label = rf"{itype} #{i + 1}"
                    detail = {
                        "anomaly": label,
                        "idx": i,
                        "tkey": tkey,
                        "df": idf,
                        "pupitre": (df_pupitre, pupitre_dict.get(housing, {}).get(key)),
                        "archive": (df_archive, channels_dict.get(key)),
                    }
                    manager.add(fig, 0, t_mid, f_mid, label, detail)

        if df_hybrid_incidents is not None:
            hy_ch = (hybrid_dict or {}).get(housing, {}).get(key)
            logger.info(
                f"Adding hybrid incident annotations: {list(df_hybrid_incidents.keys())}"
            )
            for itype, incident_list in df_hybrid_incidents.items():
                for i, idf in enumerate(incident_list):
                    if idf.empty:
                        continue
                    t_mid = idf[tkey].median()
                    if hy_ch is None or hy_ch not in idf.columns:
                        continue
                    f_mid = idf[hy_ch].median()
                    label = rf"{itype} #{i + 1}"
                    detail = {
                        "anomaly": label,
                        "idx": i,
                        "tkey": tkey,
                        "df": idf,
                        "hybrid": (df_hybrid, hy_ch),
                        "archive": (df_archive, channels_dict.get(key)),
                    }
                    manager.add(fig, 0, t_mid, f_mid, label, detail)

        manager.connect(fig)

    if not _direct_mpl:
        b.finalize(fig)

    if save:
        if output_path is None:
            _lbl, igroup = key.split("_") if "_" in key else (key, "")
            output_path = f"{title.replace('_Overview', '')}-{igroup}.png"
        b.save(fig, Path(output_path), dpi=style.dpi)
        logger.info(f"Saved plot to {output_path}")

    if show:
        b.show(fig)
        return None

    return fig


def plot_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    x_col: str,
    y_col1: str,
    y_col2: str,
    label1: str = "Series 1",
    label2: str = "Series 2",
    title: str = "Comparison",
    downsample_percent: float = 100.0,
    show: bool = False,
    save: bool = False,
    output_path: str | None = None,
    style: PlotStyle | None = None,
    backend: str | PlottingBackend = "matplotlib",
) -> Any | None:
    """
    Plot comparison between two DataFrames.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to compare
    x_col : str
        X-axis column name
    y_col1, y_col2 : str
        Y-axis column names in each DataFrame
    label1, label2 : str
        Labels for legend
    title : str
        Plot title
    downsample_percent : float
        Percentage of data to plot (converted to DownsampleConfig internally)
    show : bool
        Display plot
    save : bool
        Save plot to file
    output_path : str, optional
        Output file path
    style : PlotStyle, optional
        Plot style configuration
    backend : str or PlottingBackend, optional
        Backend name or instance (default: ``"matplotlib"``).

    Returns
    -------
    Figure or None
    """
    style = style or DEFAULT_STYLE
    b = get_backend(backend)

    ds_config: DownsampleConfig | None = None
    if downsample_percent < 100.0:
        n_points = max(len(df1), len(df2))
        ds_config = DownsampleConfig(
            n_out=max(100, int(n_points * downsample_percent / 100.0))
        )

    # Rename y columns to their legend labels, concat along the shared x axis.
    s1 = df1[[x_col, y_col1]].rename(columns={y_col1: label1})
    s2 = df2[[x_col, y_col2]].rename(columns={y_col2: label2})
    merged = (
        pd.concat([s1, s2], ignore_index=True)
        .sort_values(x_col, kind="stable")
        .reset_index(drop=True)
    )

    fig = plot_overlay(
        merged,
        [label1, label2],
        t_col=x_col,
        backend=b,
        style=style,
        title=title,
        colors=["blue", "red"],
        downsample=ds_config,
    )
    fig._magnetrun_xlabel = x_col  # type: ignore[attr-defined]
    b.finalize(fig)

    if save and output_path:
        b.save(fig, Path(output_path), dpi=style.dpi)

    if show:
        b.show(fig)
        return None

    return fig


def plot_regimes(
    ax: Any,
    regimes: list[str],
    times: list[float],
    colors: PlotColors | None = None,
    alpha: float = 0.2,
) -> None:
    """
    Add regime spans to a plot.

    Highlights different operational regimes (Up, Down, Plateau) with
    colored vertical spans.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add spans to
    regimes : List[str]
        List of regime types ('U', 'D', 'P')
    times : List[float]
        List of transition times
    colors : PlotColors, optional
        Color configuration
    alpha : float, optional
        Transparency for spans

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(t, y)
    >>> plot_regimes(ax, ['U', 'P', 'D'], [0, 10, 50, 100])
    """
    colors = colors or DEFAULT_COLORS

    if len(times) < 2:
        return

    t0 = times[0]
    for i in range(1, min(len(regimes) + 1, len(times))):
        regime = regimes[i - 1] if i - 1 < len(regimes) else "P"
        color = colors.get_regime_color(regime)
        ax.axvspan(t0, times[i], facecolor=color, alpha=alpha)
        t0 = times[i]


def plot_incidents_markers(
    ax: Any,
    incident_times: list[float],
    color: str = "red",
    alpha: float = 0.3,
    linestyle: str = "--",
) -> None:
    """
    Add vertical lines at incident times.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add markers to
    incident_times : List[float]
        Times of incidents
    color : str
        Line color
    alpha : float
        Line transparency
    linestyle : str
        Line style
    """
    for t in incident_times:
        ax.axvline(t, color=color, alpha=alpha, linestyle=linestyle)


def plot_time_series(
    df: pd.DataFrame,
    x_col: str,
    y_cols: str | list[str],
    title: str = "",
    downsample_percent: float = 100.0,
    normalize: bool = False,
    show: bool = False,
    save: bool = False,
    output_path: str | None = None,
    style: PlotStyle | None = None,
    backend: str | PlottingBackend = "matplotlib",
) -> Any | None:
    """
    Plot one or more time series from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot
    x_col : str
        X-axis column
    y_cols : str or List[str]
        Y-axis column(s)
    title : str
        Plot title
    downsample_percent : float
        Percentage of data to plot (converted to DownsampleConfig internally)
    normalize : bool
        Normalize each series to [0, 1]
    show : bool
        Display plot
    save : bool
        Save plot to file
    output_path : str, optional
        Output file path
    style : PlotStyle, optional
        Plot style configuration
    backend : str or PlottingBackend, optional
        Backend name or instance (default: ``"matplotlib"``).

    Returns
    -------
    Figure or None
    """
    style = style or DEFAULT_STYLE
    b = get_backend(backend)

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    ds_config: DownsampleConfig | None = None
    if downsample_percent < 100.0:
        ds_config = DownsampleConfig(
            n_out=max(100, int(len(df) * downsample_percent / 100.0))
        )

    fig = plot_overlay(
        df,
        list(y_cols),
        t_col=x_col,
        backend=b,
        style=style,
        title=title,
        normalize=normalize,
        downsample=ds_config,
    )
    fig._magnetrun_xlabel = x_col  # type: ignore[attr-defined]
    b.finalize(fig)

    if save and output_path:
        b.save(fig, Path(output_path), dpi=style.dpi)

    if show:
        b.show(fig)
        return None

    return fig


# =============================================================================
# Utility functions
# =============================================================================
def setup_matplotlib_defaults() -> None:
    """
    Setup matplotlib with sensible defaults for scientific plotting.

    Enables LaTeX rendering if available and sets common style options.
    """

    try:
        matplotlib.rcParams["text.usetex"] = True
    except (OSError, RuntimeError):
        logger.debug("LaTeX not available for matplotlib")

    plt.style.use("seaborn-v0_8-whitegrid")


def create_figure_grid(
    n_plots: int,
    n_cols: int = 2,
    figsize_per_plot: tuple[float, float] = (6, 4),
) -> tuple[Any, np.ndarray]:
    """
    Create a grid of subplots.

    Parameters
    ----------
    n_plots : int
        Number of plots needed
    n_cols : int
        Number of columns in grid
    figsize_per_plot : tuple
        Size of each subplot

    Returns
    -------
    Tuple[Figure, np.ndarray]
        Figure and array of axes
    """

    n_rows = (n_plots + n_cols - 1) // n_cols
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is always 2D array
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes


def save_figure(
    fig: Any,
    output_path: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **kwargs,
) -> None:
    """
    Save a figure with common settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    output_path : str
        Output file path
    dpi : int
        Resolution
    bbox_inches : str
        Bounding box setting
    **kwargs
        Additional arguments to savefig
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    logger.info(f"Saved figure to {output_path}")
