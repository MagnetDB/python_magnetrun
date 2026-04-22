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
        downsample_for_plot,
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
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python_magnetrun.plotting.style import (  # noqa: F401
    DEFAULT_COLORS,
    DEFAULT_STYLE,
    PlotColors,
    PlotStyle,
)
from python_magnetrun.utils.downsampling import DownsampleConfig, downsample_arrays

# Module logger
logger = logging.getLogger("python_magnetrun.analysis.plotting")


# =============================================================================
# Downsampling functions
# =============================================================================
def downsample_for_plot(
    x: np.ndarray,
    y: np.ndarray,
    percent: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample data for plotting to reduce memory usage and rendering time.

    This is essential for plotting high-frequency data (e.g., 4800 Hz incident
    files) where millions of points would overwhelm the plotting system.

    Parameters
    ----------
    x : np.ndarray
        X-axis data (e.g., time)
    y : np.ndarray
        Y-axis data (e.g., values)
    percent : float, optional
        Percentage of points to keep (0-100). Default is 100 (no downsampling).
        A value of 10 means keep 10% of the data points.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Downsampled (x, y) arrays

    Examples
    --------
    >>> x = np.arange(1000000)
    >>> y = np.sin(x / 1000)
    >>> x_ds, y_ds = downsample_for_plot(x, y, percent=1.0)
    >>> len(x_ds)  # ~10000 points
    10000

    Notes
    -----
    Uses uniform step-based downsampling which preserves the overall shape
    but may miss sharp peaks. For peak-preserving downsampling, consider
    using `downsample_lttb` or `downsample_minmax`.
    """
    if percent >= 100.0:
        return x, y

    if percent <= 0.0:
        logger.warning("downsample percent <= 0, returning single point")
        return x[:1], y[:1]

    n = len(x)
    n_keep = max(1, int(n * percent / 100.0))
    step = max(1, n // n_keep)

    logger.debug(
        f"Downsampling: {n} -> {len(x[::step])} points ({percent:.1f}%, step={step})"
    )

    return x[::step], y[::step]


def downsample_dataframe(
    df: pd.DataFrame,
    percent: float = 100.0,
    preserve_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Downsample a DataFrame for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to downsample
    percent : float, optional
        Percentage of rows to keep (0-100)
    preserve_columns : List[str], optional
        Columns that must be included (not affected by downsampling logic)

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame
    """
    if percent >= 100.0:
        return df

    n = len(df)
    n_keep = max(1, int(n * percent / 100.0))
    step = max(1, n // n_keep)

    return df.iloc[::step].copy()


def downsample_minmax(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample preserving min/max values in each bin.

    This method divides the data into bins and keeps both the minimum
    and maximum value in each bin, preserving peaks and valleys.

    Parameters
    ----------
    x : np.ndarray
        X-axis data
    y : np.ndarray
        Y-axis data
    n_bins : int, optional
        Number of bins to divide data into

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Downsampled (x, y) arrays with preserved extrema
    """
    n = len(x)
    if n <= n_bins * 2:
        return x, y

    bin_size = n // n_bins
    x_out = []
    y_out = []

    for i in range(n_bins):
        start = i * bin_size
        end = min(start + bin_size, n)

        y_slice = y[start:end]
        x_slice = x[start:end]

        min_idx = np.argmin(y_slice)
        max_idx = np.argmax(y_slice)

        # Add in order (min first if it comes before max)
        if min_idx <= max_idx:
            x_out.extend([x_slice[min_idx], x_slice[max_idx]])
            y_out.extend([y_slice[min_idx], y_slice[max_idx]])
        else:
            x_out.extend([x_slice[max_idx], x_slice[min_idx]])
            y_out.extend([y_slice[max_idx], y_slice[min_idx]])

    return np.array(x_out), np.array(y_out)


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
    downsample_config: DownsampleConfig | None = None,
    style: PlotStyle | None = None,
    colors: PlotColors | None = None,
    interactive: bool = True,
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

    # Create figure
    fig, ax = plt.subplots(figsize=style.figsize)
    legends = []

    # Helper to downsample and plot
    def plot_series(df, x_col, y_col, color, label, marker=None, alpha=1.0):
        if df is None or df.empty:
            return

        x = df[x_col].values
        y = df[y_col].values

        if downsample_config is not None:
            y, x = downsample_arrays(y, x, downsample_config)

        if marker:
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                alpha=alpha,
                markersize=3,
                linestyle="-",
            )
        else:
            ax.plot(x, y, color=color, alpha=alpha)
        legends.append(label)

    # Plot overview data
    if not df_overview.empty:
        plot_series(df_overview, tkey, key, colors.overview, f"Overview: {key}")
        plot_series(
            df_overview,
            tkey,
            channels_dict[key],
            colors.overview,
            f"Overview: {channels_dict[key]}",
            marker=".",
            alpha=0.7,
        )

    # Plot archive data
    if not df_archive.empty:
        plot_series(
            df_archive,
            tkey,
            channels_dict[key],
            colors.archive,
            f"Archive: {channels_dict[key]}",
            alpha=0.5,
        )

    # Plot pupitre data
    if not df_pupitre.empty:
        logger.info(
            f"housing={housing}, key={key}, pupitre_dict={pupitre_dict}, "
            f"df.columns={df_pupitre.columns.tolist()}"
        )
        pupitre_key = pupitre_dict[housing][key]
        plot_series(
            df_pupitre, tkey, pupitre_key, colors.pupitre, f"Pupitre: {pupitre_key}"
        )

    # Plot incidents via AnnotationManager
    if df_incidents is not None and interactive:
        from python_magnetrun.plotting.annotations import AnnotationManager
        from python_magnetrun.plotting.matplotlib_backend import MatplotlibBackend

        manager = AnnotationManager(MatplotlibBackend(), style=style, colors=colors)

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

        manager.connect(fig)

    # Finalize plot
    ax.legend(labels=legends, loc=style.legend_loc)
    ax.set_title(
        f"{title.replace('_Overview', '')}: {key} {msg}", fontsize=style.title_fontsize
    )
    ax.set_xlabel(tkey, fontsize=style.label_fontsize)

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    plt.tight_layout()

    # Save if requested
    if save:
        if output_path is None:
            # Auto-generate filename
            label, igroup = key.split("_") if "_" in key else (key, "")
            output_path = f"{title.replace('_Overview', '')}-{igroup}.png"
        plt.savefig(output_path, dpi=style.dpi)
        logger.info(f"Saved plot to {output_path}")

    if show:
        plt.show()
        plt.close()
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
        Percentage of data to plot
    show : bool
        Display plot
    save : bool
        Save plot to file
    output_path : str, optional
        Output file path
    style : PlotStyle, optional
        Plot style configuration

    Returns
    -------
    Figure or None
    """

    style = style or DEFAULT_STYLE

    fig, ax = plt.subplots(figsize=style.figsize)

    # Plot first series
    x1, y1 = df1[x_col].to_numpy(), df1[y_col1].to_numpy()
    if downsample_percent < 100.0:
        x1, y1 = downsample_for_plot(x1, y1, downsample_percent)
    ax.plot(x1, y1, label=label1, color="blue")

    # Plot second series
    x2, y2 = df2[x_col].to_numpy(), df2[y_col2].to_numpy()
    if downsample_percent < 100.0:
        x2, y2 = downsample_for_plot(x2, y2, downsample_percent)
    ax.plot(x2, y2, label=label2, color="red", alpha=0.7)

    ax.set_xlabel(x_col)
    ax.set_title(title)
    ax.legend()

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    plt.tight_layout()

    if save and output_path:
        plt.savefig(output_path, dpi=style.dpi)

    if show:
        plt.show()
        plt.close()
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
        Percentage of data to plot
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

    Returns
    -------
    Figure or None
    """

    style = style or DEFAULT_STYLE

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    fig, ax = plt.subplots(figsize=style.figsize)

    x = df[x_col].to_numpy()

    for y_col in y_cols:
        y = df[y_col].to_numpy()

        if normalize:
            y_min, y_max = y.min(), y.max()
            if y_max - y_min > 0:
                y = (y - y_min) / (y_max - y_min)

        if downsample_percent < 100.0:
            x_plot, y_plot = downsample_for_plot(x, y, downsample_percent)
        else:
            x_plot, y_plot = x, y

        ax.plot(x_plot, y_plot, label=y_col)

    ax.set_xlabel(x_col)
    ax.set_ylabel("Normalized" if normalize else "Value")
    ax.set_title(title)
    ax.legend()

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    plt.tight_layout()

    if save and output_path:
        plt.savefig(output_path, dpi=style.dpi)

    if show:
        plt.show()
        plt.close()
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
