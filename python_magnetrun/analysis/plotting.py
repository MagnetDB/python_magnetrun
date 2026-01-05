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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Module logger
logger = logging.getLogger("magnetrun.analysis.plotting")


# =============================================================================
# Configuration dataclasses
# =============================================================================
@dataclass
class PlotStyle:
    """
    Configuration for plot styling.
    
    Attributes
    ----------
    figsize : Tuple[int, int]
        Figure size in inches (width, height)
    dpi : int
        Resolution for saved figures
    grid : bool
        Whether to show grid
    grid_alpha : float
        Grid transparency
    legend_loc : str
        Legend location
    title_fontsize : int
        Title font size
    label_fontsize : int
        Axis label font size
    """
    
    figsize: Tuple[int, int] = (12, 5)
    dpi: int = 300
    grid: bool = True
    grid_alpha: float = 0.3
    legend_loc: str = "best"
    title_fontsize: int = 12
    label_fontsize: int = 10


@dataclass
class PlotColors:
    """
    Color configuration for different data sources and regimes.
    
    Attributes
    ----------
    overview : str
        Color for overview data
    archive : str
        Color for archive data
    pupitre : str
        Color for pupitre data
    incident : str
        Color for incident markers
    regime_up : str
        Color for 'Up' regime spans
    regime_down : str
        Color for 'Down' regime spans
    regime_plateau : str
        Color for 'Plateau' regime spans
    """
    
    overview: str = "blue"
    archive: str = "red"
    pupitre: str = "green"
    incident: str = "yellow"
    regime_up: str = "green"
    regime_down: str = "red"
    regime_plateau: str = "blue"
    
    def get_regime_color(self, regime: str) -> str:
        """Get color for a regime type."""
        regime_map = {
            "U": self.regime_up,
            "D": self.regime_down,
            "P": self.regime_plateau,
        }
        return regime_map.get(regime, "gray")


# Default instances
DEFAULT_STYLE = PlotStyle()
DEFAULT_COLORS = PlotColors()


# =============================================================================
# Downsampling functions
# =============================================================================
def downsample_for_plot(
    x: np.ndarray,
    y: np.ndarray,
    percent: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
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
        "Downsampling: %d -> %d points (%.1f%%, step=%d)",
        n, len(x[::step]), percent, step
    )
    
    return x[::step], y[::step]


def downsample_dataframe(
    df: pd.DataFrame,
    percent: float = 100.0,
    preserve_columns: Optional[List[str]] = None,
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
) -> Tuple[np.ndarray, np.ndarray]:
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
    df_incidents: Optional[Dict[str, List[pd.DataFrame]]],
    channels_dict: Dict[str, str],
    pupitre_dict: Dict[str, Dict[str, str]],
    site: str,
    tkey: str,
    key: str,
    title: str,
    msg: str,
    show: bool = False,
    save: bool = False,
    output_path: Optional[str] = None,
    downsample_percent: float = 100.0,
    style: Optional[PlotStyle] = None,
    colors: Optional[PlotColors] = None,
    interactive: bool = True,
) -> Optional[Any]:
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
        Site-specific mapping for pupitre channels
    site : str
        Site identifier (e.g., "M9")
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
    downsample_percent : float, optional
        Percentage of data to plot (0-100). Default is 100 (no downsampling).
        Use lower values for large datasets.
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
    ...     downsample_percent=10.0,  # Plot only 10% of points
    ...     show=True,
    ... )
    """
    import matplotlib.pyplot as plt
    
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
        
        if downsample_percent < 100.0:
            x, y = downsample_for_plot(x, y, downsample_percent)
        
        if marker:
            ax.plot(x, y, color=color, marker=marker, alpha=alpha, markersize=3, linestyle='-')
        else:
            ax.plot(x, y, color=color, alpha=alpha)
        legends.append(label)
    
    # Plot overview data
    if not df_overview.empty:
        plot_series(df_overview, tkey, key, colors.overview, f"Overview: {key}")
        plot_series(
            df_overview, tkey, channels_dict[key],
            colors.overview, f"Overview: {channels_dict[key]}",
            marker=".", alpha=0.7
        )
    
    # Plot archive data
    if not df_archive.empty:
        plot_series(
            df_archive, tkey, channels_dict[key],
            colors.archive, f"Archive: {channels_dict[key]}",
            alpha=0.5
        )
    
    # Plot pupitre data
    if not df_pupitre.empty:
        pupitre_key = pupitre_dict[site][key]
        plot_series(
            df_pupitre, tkey, pupitre_key,
            colors.pupitre, f"Pupitre: {pupitre_key}"
        )
    
    # Store annotation metadata for interactivity
    annotation_dict = {}
    
    # Plot incidents
    if df_incidents is not None and interactive:
        for itype, incident_list in df_incidents.items():
            for i, idf in enumerate(incident_list):
                if idf.empty:
                    continue
                
                # Get midpoint for annotation
                t_mid = idf[tkey].median()
                
                # Get the channel key for this incident
                incident_key = channels_dict.get(key, key)
                if incident_key not in idf.columns:
                    continue
                
                f_mid = idf[incident_key].median()
                
                # Plot marker
                point, = ax.plot(t_mid, f_mid, "yo", markersize=8)
                
                # Add annotation with arrow
                annot = ax.annotate(
                    rf"{itype} \#{i+1}",
                    xy=(t_mid, f_mid),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc=colors.incident, alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )
                
                # Make annotation clickable
                annot.set_picker(True)
                
                # Store metadata for click handler
                annotation_dict[annot] = {
                    "anomaly": rf"{itype} \#{i+1}",
                    "idx": i,
                    "df": idf,
                    "pupitre": (df_pupitre, pupitre_dict.get(site, {}).get(key)),
                    "archive": (df_archive, channels_dict.get(key)),
                }
    
    # Setup interactive click handler
    if interactive and annotation_dict:
        open_figures = {}
        
        def on_pick(event):
            if event.artist in annotation_dict:
                metadata = annotation_dict[event.artist]
                _show_incident_detail(
                    metadata, open_figures, tkey,
                    colors, style, downsample_percent
                )
        
        fig.canvas.mpl_connect("pick_event", on_pick)
    
    # Finalize plot
    ax.legend(labels=legends, loc=style.legend_loc)
    ax.set_title(f'{title.replace("_Overview", "")}: {key} {msg}', fontsize=style.title_fontsize)
    ax.set_xlabel(tkey, fontsize=style.label_fontsize)
    
    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        if output_path is None:
            # Auto-generate filename
            label, igroup = key.split("_") if "_" in key else (key, "")
            output_path = f'{title.replace("_Overview", "")}-{igroup}.png'
        plt.savefig(output_path, dpi=style.dpi)
        logger.info("Saved plot to %s", output_path)
    
    if show:
        plt.show()
        plt.close()
        return None
    
    return fig


def _show_incident_detail(
    metadata: Dict[str, Any],
    open_figures: Dict[int, Any],
    tkey: str,
    colors: PlotColors,
    style: PlotStyle,
    downsample_percent: float,
) -> None:
    """Show detailed view of an incident when clicked."""
    import matplotlib.pyplot as plt
    
    anomaly = metadata["anomaly"]
    idx = metadata["idx"]
    idf = metadata["df"]
    pupitre, pupitre_key = metadata["pupitre"]
    archive, archive_key = metadata["archive"]
    
    # Close previous subplot if it exists
    if idx in open_figures:
        plt.close(open_figures[idx])
    
    # Create subplot
    fig_sub, ax_sub = plt.subplots(figsize=(8, 5))
    
    # Plot incident data
    if archive_key and archive_key in idf.columns:
        x, y = idf[tkey].values, idf[archive_key].values
        if downsample_percent < 100.0:
            x, y = downsample_for_plot(x, y, downsample_percent)
        ax_sub.plot(x, y, color=colors.incident, label=f"Incident: {archive_key}")
    
    # Plot pupitre context
    if pupitre is not None and not pupitre.empty and pupitre_key:
        t_start, t_end = idf[tkey].iloc[0], idf[tkey].iloc[-1]
        mask = (pupitre[tkey] >= t_start) & (pupitre[tkey] <= t_end)
        pupitre_slice = pupitre[mask]
        if not pupitre_slice.empty and pupitre_key in pupitre_slice.columns:
            x, y = pupitre_slice[tkey].values, pupitre_slice[pupitre_key].values
            ax_sub.plot(x, y, color=colors.pupitre, alpha=0.7, label=f"Pupitre: {pupitre_key}")
    
    # Plot archive context
    if archive is not None and not archive.empty and archive_key:
        t_start, t_end = idf[tkey].iloc[0], idf[tkey].iloc[-1]
        mask = (archive[tkey] >= t_start) & (archive[tkey] <= t_end)
        archive_slice = archive[mask]
        if not archive_slice.empty and archive_key in archive_slice.columns:
            x, y = archive_slice[tkey].values, archive_slice[archive_key].values
            ax_sub.plot(x, y, color=colors.archive, alpha=0.5, label=f"Archive: {archive_key}")
    
    ax_sub.set_xlabel(tkey)
    ax_sub.set_xlim(idf[tkey].iloc[0], idf[tkey].iloc[-1])
    ax_sub.set_title(anomaly)
    ax_sub.legend()
    ax_sub.grid(True, alpha=style.grid_alpha)
    
    fig_sub.tight_layout()
    fig_sub.show()
    
    # Store figure reference
    open_figures[idx] = fig_sub


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
    output_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Any]:
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
    import matplotlib.pyplot as plt
    
    style = style or DEFAULT_STYLE
    
    fig, ax = plt.subplots(figsize=style.figsize)
    
    # Plot first series
    x1, y1 = df1[x_col].values, df1[y_col1].values
    if downsample_percent < 100.0:
        x1, y1 = downsample_for_plot(x1, y1, downsample_percent)
    ax.plot(x1, y1, label=label1, color="blue")
    
    # Plot second series
    x2, y2 = df2[x_col].values, df2[y_col2].values
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
    regimes: List[str],
    times: List[float],
    colors: Optional[PlotColors] = None,
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
    incident_times: List[float],
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
    y_cols: Union[str, List[str]],
    title: str = "",
    downsample_percent: float = 100.0,
    normalize: bool = False,
    show: bool = False,
    save: bool = False,
    output_path: Optional[str] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Any]:
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
    import matplotlib.pyplot as plt
    
    style = style or DEFAULT_STYLE
    
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    
    fig, ax = plt.subplots(figsize=style.figsize)
    
    x = df[x_col].values
    
    for y_col in y_cols:
        y = df[y_col].values
        
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
    import matplotlib
    import matplotlib.pyplot as plt
    
    try:
        matplotlib.rcParams["text.usetex"] = True
    except Exception:
        logger.debug("LaTeX not available for matplotlib")
    
    plt.style.use("seaborn-v0_8-whitegrid")


def create_figure_grid(
    n_plots: int,
    n_cols: int = 2,
    figsize_per_plot: Tuple[float, float] = (6, 4),
) -> Tuple[Any, np.ndarray]:
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
    import matplotlib.pyplot as plt
    
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
    logger.info("Saved figure to %s", output_path)
