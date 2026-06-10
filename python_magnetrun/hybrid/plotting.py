"""
Plotting functions for hybrid data visualization

This module provides plotting functions for kHz, RMS, and combined data.
Functions accept a HybridData instance and variable specifications.

Single variable functions:
- plot_khz_variable(): Plot a single kHz variable
- plot_rms_variable(): Plot a single RMS variable
- plot_khz_with_rms(): Plot kHz and RMS data together

Multi-variable functions:
- plot_khz_variables(): Plot multiple kHz variables (subplots or overlay)
- plot_rms_variables(): Plot multiple RMS variables (subplots or overlay)

Performance features:
- Automatic downsampling for large datasets using tsdownsample (MinMaxLTTB)
- Preserves visual fidelity (peaks/valleys) while reducing data points
- Set downsample parameter to target number of points for plotting

Note:
- Outlier detection is now handled by the separate `python_magnetrun.outliers` module
- For outlier analysis, use `from python_magnetrun.outliers import OutlierDetector, detect_outliers`
- Plotting functions accept pre-computed outlier masks via `outlier_mask` parameter
"""

import logging
import struct
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from python_magnetrun.plotting.backend import PlottingBackend, get_backend
from python_magnetrun.plotting.timeseries import plot_overlay, plot_subplots
from python_magnetrun.utils.downsampling import DownsampleConfig, downsample_arrays

# Type checking import to avoid circular import
if TYPE_CHECKING:
    from ..outliers import OutlierResult
    from .hybrid_data import HybridData

# Import outlier detection from dedicated module
from ..outliers import OutlierResult

# Setup logger
logger = logging.getLogger(__name__)


def _make_downsample_config(target_points: int, method: str) -> DownsampleConfig:
    """Map hybrid plot method names to DownsampleConfig."""
    ds_method = "minmax_lttb" if method == "auto" else method
    return DownsampleConfig(n_out=target_points, method=ds_method)


def _get_khz_unit(hybrid_data: "HybridData", system: str, variable: str) -> str:
    """Get unit for a kHz variable from config."""
    config = hybrid_data.load_khz_config(system)
    if config is None:
        return ""

    for card in config.cards:
        if variable in card.variable_names:
            idx = card.variable_names.index(variable)
            if (
                card.calibrations
                and idx < len(card.calibrations)
                and card.calibrations[idx]
            ):
                return card.calibrations[idx].unit or ""
    return ""


def _get_rms_unit(
    hybrid_data: "HybridData", system: str, variable: str, file_idx: int = 0
) -> str:
    """Get unit for an RMS variable from variable info."""
    try:
        var_info = hybrid_data.get_rms_variable_info(system, file_idx=file_idx)
        var_row = var_info[var_info["name"] == variable]
        if not var_row.empty and var_row.iloc[0]["unit"]:
            return var_row.iloc[0]["unit"]
    except (OSError, ValueError, RuntimeError, KeyError):
        pass
    return ""


def _handle_output(
    b: PlottingBackend,
    fig: Any,
    show: bool,
    save: str | None,
) -> None:
    b.finalize(fig, xlabel="Time (s)")
    if save:
        from pathlib import Path
        b.save(fig, Path(save))
    if show:
        b.show(fig)


def _apply_outlier_strategy(
    data: np.ndarray,
    time: np.ndarray,
    outlier_result: "OutlierResult",
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply non-highlight outlier strategy, returning cleaned (data, time)."""
    data = outlier_result.apply_to_data(data, strategy=strategy)
    if strategy in ("remove", "nan"):
        valid = ~np.isnan(data)
        data = data[valid]
        time = time[valid]
    logger.info(
        "Applied %s to %d outliers", strategy, outlier_result.n_outliers
    )
    return data, time


def _scatter_outliers(
    b: PlottingBackend,
    fig: Any,
    ax_idx: int,
    data: np.ndarray,
    time: np.ndarray,
    outlier_result: "OutlierResult",
    label: str,
    downsample: int | None,
    downsample_method: str,
) -> None:
    """Scatter-plot outlier points on top of the main series."""
    mask = outlier_result.mask
    outlier_data = data[mask[: len(data)]]
    outlier_time = time[mask[: len(time)]]
    if len(outlier_data) == 0:
        return
    if downsample is not None and len(outlier_data) > downsample // 10:
        outlier_data, outlier_time = downsample_arrays(
            outlier_data,
            outlier_time,
            DownsampleConfig(n_out=downsample // 10, method="stride"),
        )
    b.add_scatter(fig, ax_idx, outlier_time, outlier_data, label=label)


def _plot_variable_impl(
    data: np.ndarray,
    time: np.ndarray,
    system: str,
    variable: str,
    ylabel: str,
    title: str,
    ax_series_label: str,
    ax: Any,
    show: bool,
    save: str | None,
    outlier_result: Optional["OutlierResult"],
    outlier_strategy: str,
    downsample: int | None,
    downsample_method: str,
    backend: str | PlottingBackend,
) -> tuple:
    """Shared plotting pipeline for a single variable (kHz or RMS).

    Parameters
    ----------
    data : numpy.ndarray
        Signal values [variable units].
    time : numpy.ndarray
        Time axis [s].
    system : str
        FEPC system name (used for logging only).
    variable : str
        Variable name (used as DataFrame column and for logging).
    ylabel : str
        Y-axis label including units, e.g. ``"I_H1 [A]"``.
    title : str
        Plot title (caller builds the full title string).
    ax_series_label : str
        Series label for the ax-injection path (e.g. ``"I_H1"`` or ``"I_H1 (RMS)"``).
    ax : matplotlib.axes.Axes or None
        Pre-existing axes for injection (matplotlib only); ``None`` creates a new figure.
    show : bool
        Call backend show after plotting.
    save : str or None
        Path to save the figure; ``None`` skips saving.
    outlier_result : OutlierResult or None
        Pre-computed outlier detection result.
    outlier_strategy : str
        How to handle outliers: ``'remove'``, ``'interpolate'``, ``'highlight'``, ``'none'``.
    downsample : int or None
        Target number of points; ``None`` disables downsampling.
    downsample_method : str
        Downsampling method name (``'auto'``, ``'minmax_lttb'``, ``'stride'``, ``'minmax'``).
    backend : str or PlottingBackend
        Plotting backend.

    Returns
    -------
    tuple
        ``(fig, ax)`` — ``ax`` is ``None`` for non-matplotlib backends.
    """
    orig_data, orig_time = data, time
    highlight = outlier_result is not None and outlier_strategy == "highlight"

    if outlier_result is not None and outlier_strategy not in ("none", "highlight"):
        data, time = _apply_outlier_strategy(data, time, outlier_result, outlier_strategy)

    if downsample is not None and len(data) > downsample:
        orig_len = len(data)
        data, time = downsample_arrays(
            data, time, _make_downsample_config(downsample, downsample_method)
        )
        logger.info("Downsampled %s: %d -> %d points", variable, orig_len, len(data))

    if ax is not None:
        fig = ax.get_figure()
        fig._magnetrun_axes = [ax]
        b = get_backend("matplotlib")
    else:
        b = get_backend(backend)
        df = pd.DataFrame({"t": time, variable: data})
        fig = plot_overlay(df, [variable], t_col="t", backend=b, title=title)
        if ylabel != variable and hasattr(b, "_get_ax"):
            b._get_ax(fig, 0).set_ylabel(ylabel)

        if highlight and outlier_result is not None:
            _scatter_outliers(
                b, fig, 0, orig_data, orig_time, outlier_result,
                f"Outliers ({outlier_result.n_outliers:,})",
                downsample, downsample_method,
            )

        _handle_output(b, fig, show, save)
        out_ax = fig._magnetrun_axes[0] if hasattr(fig, "_magnetrun_axes") else None
        return fig, out_ax

    # ax-injection path (matplotlib only).
    b.add_series(fig, 0, time, data, label=ax_series_label)
    if highlight and outlier_result is not None:
        _scatter_outliers(
            b, fig, 0, orig_data, orig_time, outlier_result,
            f"Outliers ({outlier_result.n_outliers:,})",
            downsample, downsample_method,
        )
    _handle_output(b, fig, show, save)
    return fig, ax


def _plot_variables_impl(
    hybrid_data: "HybridData",
    system: str,
    variables: list[str],
    read_fn: Callable[[str], tuple[np.ndarray, np.ndarray]],
    get_unit_fn: Callable[[str], str],
    title: str,
    layout: str,
    show: bool,
    save: str | None,
    outlier_results: dict[str, "OutlierResult"] | None,
    outlier_strategy: str,
    downsample: int | None,
    downsample_method: str,
    backend: str | PlottingBackend,
) -> tuple:
    """Shared plotting pipeline for multiple variables (kHz or RMS).

    Parameters
    ----------
    hybrid_data : HybridData
        HybridData instance (used only for :attr:`date_str` logging).
    system : str
        FEPC system name.
    variables : list of str
        Variable names to plot.
    read_fn : callable
        ``read_fn(variable) -> (data, time)`` — reads one variable's arrays.
    get_unit_fn : callable
        ``get_unit_fn(variable) -> str`` — returns the physical unit string.
    title : str
        Plot title.
    layout : str
        ``'subplots'`` or ``'overlay'``.
    show : bool
        Call backend show after plotting.
    save : str or None
        Path to save the figure; ``None`` skips saving.
    outlier_results : dict or None
        Maps variable names to :class:`~python_magnetrun.outliers.OutlierResult`.
    outlier_strategy : str
        How to handle outliers: ``'remove'``, ``'interpolate'``, ``'highlight'``, ``'none'``.
    downsample : int or None
        Target number of points; ``None`` disables downsampling.
    downsample_method : str
        Downsampling method name.
    backend : str or PlottingBackend
        Plotting backend.

    Returns
    -------
    tuple
        ``(fig, axes)`` — axes is a list for subplots, a single axes for overlay,
        or ``None`` for non-matplotlib backends.
    """
    b = get_backend(backend)

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    highlight_vars: list[str] = []
    orig_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for variable in variables:
        try:
            data, time = read_fn(variable)
            var_outlier = outlier_results.get(variable) if outlier_results else None
            if var_outlier is not None and outlier_strategy not in ("none", "highlight"):
                data, time = _apply_outlier_strategy(data, time, var_outlier, outlier_strategy)
            elif var_outlier is not None and outlier_strategy == "highlight":
                orig_series[variable] = (data.copy(), time.copy())
                highlight_vars.append(variable)

            if downsample is not None and len(data) > downsample:
                orig_len = len(data)
                data, time = downsample_arrays(
                    data, time, _make_downsample_config(downsample, downsample_method)
                )
                logger.info("Downsampled %s: %d -> %d points", variable, orig_len, len(data))

            series[variable] = (data, time)
        except (OSError, ValueError, RuntimeError, KeyError) as e:
            logger.error("Error loading %s: %s", variable, e)

    if not series:
        raise RuntimeError(f"No data loaded for {system}")

    dfs = [
        pd.DataFrame({"t": time, variable: data})
        for variable, (data, time) in series.items()
    ]
    merged = (
        pd.concat(dfs, ignore_index=True)
        .sort_values("t", kind="stable")
        .reset_index(drop=True)
    )

    valid_vars = list(series.keys())
    units = {v: get_unit_fn(v) for v in valid_vars}
    ylabel_map = {v: f"{v} [{units[v]}]" if units[v] else v for v in valid_vars}

    _plot_fn = plot_subplots if layout != "overlay" else plot_overlay
    fig = _plot_fn(merged, valid_vars, t_col="t", backend=b, title=title)

    for i, v in enumerate(valid_vars):
        if ylabel_map[v] != v and hasattr(b, "_get_ax"):
            ax = b._get_ax(fig, i if layout != "overlay" else 0)
            ax.set_ylabel(ylabel_map[v])

    for variable in highlight_vars:
        var_outlier = (outlier_results or {}).get(variable)
        if var_outlier is None:
            continue
        orig_data, orig_time = orig_series.get(variable, series[variable])
        ax_idx = valid_vars.index(variable) if layout != "overlay" else 0
        _scatter_outliers(
            b, fig, ax_idx, orig_data, orig_time, var_outlier,
            f"{variable} outliers", downsample, downsample_method,
        )

    _handle_output(b, fig, show, save)

    axes = None
    if hasattr(fig, "_magnetrun_axes"):
        axes = fig._magnetrun_axes if layout != "overlay" else fig._magnetrun_axes[0]
    return fig, axes


def plot_khz_variables(
    hybrid_data: "HybridData",
    system: str,
    variables: list[str],
    hours: list[int] | None = None,
    apply_calib: bool = True,
    cnv_dir: str | None = None,
    layout: str = "subplots",
    share_x: bool = True,
    show: bool = True,
    save: str | None = None,
    outlier_results: dict[str, "OutlierResult"] | None = None,
    outlier_strategy: str = "interpolate",
    downsample: int | None = 50000,
    downsample_method: str = "auto",
    backend: str | PlottingBackend = "matplotlib",
    **plot_kwargs,
) -> tuple:
    """
    Plot multiple kHz variables

    Parameters
    ----------
    hybrid_data : HybridData
        HybridData instance
    system : str
        FEPC system name
    variables : list of str
        List of variable names to plot
    hours : list of int, optional
        Hours to read (default: all available)
    apply_calib : bool, optional
        Apply calibration (default: True)
    cnv_dir : str, optional
        Directory for CNV calibration files
    layout : str, optional
        Plot layout: 'subplots' (default) or 'overlay'
    share_x : bool, optional
        Share x-axis in subplots layout (default: True)
    show : bool, optional
        Show plot (default: True)
    save : str, optional
        Save plot to file
    outlier_results : dict, optional
        Dictionary mapping variable names to OutlierResult objects.
    outlier_strategy : str, optional
        Strategy for handling outliers: 'remove', 'interpolate', 'highlight', 'none'
    downsample : int, optional
        Target number of points for plotting (default: 50000).
    downsample_method : str, optional
        Downsampling method: 'auto', 'minmax_lttb', 'stride', 'minmax'
    backend : str or PlottingBackend, optional
        Plotting backend (default: 'matplotlib').

    Returns
    -------
    tuple
        (fig, axes) where axes is a list for subplots, a single axes for overlay,
        or None for non-matplotlib backends.
    """
    n_vars = len(variables)
    if n_vars == 0:
        raise ValueError("At least one variable must be specified")

    if n_vars == 1:
        var_outlier = outlier_results.get(variables[0]) if outlier_results else None
        return plot_khz_variable(
            hybrid_data, system, variables[0],
            hours=hours, apply_calib=apply_calib, cnv_dir=cnv_dir,
            show=show, save=save,
            outlier_result=var_outlier, outlier_strategy=outlier_strategy,
            downsample=downsample, downsample_method=downsample_method,
            backend=backend,
        )

    logger.debug("plot_khz_variables: system=%s, variables=%s", system, variables)

    def _read(variable: str) -> tuple[np.ndarray, np.ndarray]:
        return hybrid_data.read_khz_variable(
            system, variable, hours=hours, apply_calib=apply_calib, cnv_dir=cnv_dir
        )

    title = f"{system} - kHz Data ({hybrid_data.date_str})"
    return _plot_variables_impl(
        hybrid_data, system, variables,
        _read,
        lambda v: _get_khz_unit(hybrid_data, system, v),
        title, layout, show, save,
        outlier_results, outlier_strategy,
        downsample, downsample_method, backend,
    )


def plot_rms_variables(
    hybrid_data: "HybridData",
    system: str,
    variables: list[str],
    file_idx: int | None = None,
    hours: list[int] | None = None,
    layout: str = "subplots",
    share_x: bool = True,
    show: bool = True,
    save: str | None = None,
    outlier_results: dict[str, "OutlierResult"] | None = None,
    outlier_strategy: str = "interpolate",
    downsample: int | None = None,
    downsample_method: str = "auto",
    backend: str | PlottingBackend = "matplotlib",
    **plot_kwargs,
) -> tuple:
    """
    Plot multiple RMS variables

    Parameters
    ----------
    hybrid_data : HybridData
        HybridData instance
    system : str
        FEPC system name
    variables : list of str
        List of variable names to plot
    file_idx : int, optional
        Index of RMS file to load
    hours : list of int, optional
        List of hours to load (0-23)
    layout : str, optional
        Plot layout: 'subplots' (default) or 'overlay'
    share_x : bool, optional
        Share x-axis in subplots layout (default: True)
    show : bool, optional
        Show plot (default: True)
    save : str, optional
        Save plot to file
    outlier_results : dict, optional
        Dictionary mapping variable names to OutlierResult objects.
    outlier_strategy : str, optional
        Strategy for handling outliers: 'remove', 'interpolate', 'highlight', 'none'
    downsample : int, optional
        Target number of points for plotting (default: None).
    downsample_method : str, optional
        Downsampling method: 'auto', 'minmax_lttb', 'stride', 'minmax'
    backend : str or PlottingBackend, optional
        Plotting backend (default: 'matplotlib').

    Returns
    -------
    tuple
        (fig, axes)
    """
    n_vars = len(variables)
    if n_vars == 0:
        raise ValueError("At least one variable must be specified")

    if n_vars == 1:
        var_outlier = outlier_results.get(variables[0]) if outlier_results else None
        return plot_rms_variable(
            hybrid_data, system, variables[0],
            file_idx=file_idx, hours=hours,
            show=show, save=save,
            outlier_result=var_outlier, outlier_strategy=outlier_strategy,
            downsample=downsample, downsample_method=downsample_method,
            backend=backend,
        )

    logger.debug("plot_rms_variables: system=%s, variables=%s", system, variables)

    info_idx = file_idx if file_idx is not None else 0

    def _read(variable: str) -> tuple[np.ndarray, np.ndarray]:
        return hybrid_data.read_rms_variable(
            system, variable, file_idx=file_idx, hours=hours
        )

    title = f"{system} - RMS Data ({hybrid_data.date_str})"
    return _plot_variables_impl(
        hybrid_data, system, variables,
        _read,
        lambda v: _get_rms_unit(hybrid_data, system, v, info_idx),
        title, layout, show, save,
        outlier_results, outlier_strategy,
        downsample, downsample_method, backend,
    )


def plot_khz_variable(
    hybrid_data: "HybridData",
    system: str,
    variable: str,
    hours: list[int] | None = None,
    apply_calib: bool = True,
    cnv_dir: str | None = None,
    ax=None,
    show: bool = True,
    save: str | None = None,
    outlier_result: Optional["OutlierResult"] = None,
    outlier_strategy: str = "interpolate",
    downsample: int | None = 50000,
    downsample_method: str = "auto",
    backend: str | PlottingBackend = "matplotlib",
    **plot_kwargs,
) -> tuple:
    """
    Plot kHz data for a specific variable

    Parameters
    ----------
    hybrid_data : HybridData
        HybridData instance
    system : str
        FEPC system name
    variable : str
        Variable name
    hours : list of int, optional
        Hours to read (default: all available)
    apply_calib : bool, optional
        Apply calibration (default: True)
    cnv_dir : str, optional
        Directory for CNV calibration files
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on (matplotlib only; ignored for other backends)
    show : bool, optional
        Show plot (default: True)
    save : str, optional
        Save plot to file
    outlier_result : OutlierResult, optional
        Pre-computed outlier detection result.
    outlier_strategy : str, optional
        How to handle outliers: 'remove', 'interpolate', 'highlight', 'none'
    downsample : int, optional
        Target number of points for plotting (default: 50000).
    downsample_method : str, optional
        Downsampling method: 'auto', 'minmax_lttb', 'stride', 'minmax'
    backend : str or PlottingBackend, optional
        Plotting backend (default: 'matplotlib').

    Returns
    -------
    tuple
        (fig, ax) — ax is None for non-matplotlib backends.
    """
    logger.debug("plot_khz_variable: system=%s, variable=%s", system, variable)

    data, time = hybrid_data.read_khz_variable(
        system, variable, hours=hours, apply_calib=apply_calib, cnv_dir=cnv_dir
    )

    config = hybrid_data.load_khz_config(system)
    if config is None:
        raise ValueError(f"No configuration found for {system}")

    unit = _get_khz_unit(hybrid_data, system, variable)
    ylabel = f"{variable} [{unit}]" if unit else variable
    title = f"{system} - {variable} ({hybrid_data.date_str})"

    return _plot_variable_impl(
        data, time, system, variable, ylabel, title, variable,
        ax, show, save, outlier_result, outlier_strategy,
        downsample, downsample_method, backend,
    )


def plot_rms_variable(
    hybrid_data: "HybridData",
    system: str,
    variable: str,
    file_idx: int | None = None,
    hours: list[int] | None = None,
    ax=None,
    show: bool = True,
    save: str | None = None,
    outlier_result: Optional["OutlierResult"] = None,
    outlier_strategy: str = "interpolate",
    downsample: int | None = None,
    downsample_method: str = "auto",
    backend: str | PlottingBackend = "matplotlib",
    **plot_kwargs,
) -> tuple:
    """
    Plot RMS data for a specific variable

    Parameters
    ----------
    hybrid_data : HybridData
        HybridData instance
    system : str
        FEPC system name
    variable : str
        Variable name
    file_idx : int, optional
        Index of RMS file to load
    hours : list of int, optional
        List of hours to load (0-23)
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on (matplotlib only)
    show : bool, optional
        Show plot (default: True)
    save : str, optional
        Save plot to file
    outlier_result : OutlierResult, optional
        Pre-computed outlier detection result.
    outlier_strategy : str, optional
        How to handle outliers: 'remove', 'interpolate', 'highlight', 'none'
    downsample : int, optional
        Target number of points for plotting.
    downsample_method : str, optional
        Downsampling method: 'auto', 'minmax_lttb', 'stride', 'minmax'
    backend : str or PlottingBackend, optional
        Plotting backend (default: 'matplotlib').

    Returns
    -------
    tuple
        (fig, ax)
    """
    logger.debug(
        "plot_rms_variable: system=%s, variable=%s, hours=%s", system, variable, hours
    )

    data, time = hybrid_data.read_rms_variable(
        system, variable, file_idx=file_idx, hours=hours
    )

    info_idx = file_idx if file_idx is not None else 0
    unit = _get_rms_unit(hybrid_data, system, variable, info_idx)
    ylabel = f"{variable} [{unit}]" if unit else f"{variable} (RMS)"
    title = f"{system} RMS - {variable} ({hybrid_data.date_str})"
    if outlier_result is not None and outlier_strategy != "none":
        title += f"\n[{outlier_strategy}: {outlier_result.n_outliers} outliers]"

    return _plot_variable_impl(
        data, time, system, variable, ylabel, title, f"{variable} (RMS)",
        ax, show, save, outlier_result, outlier_strategy,
        downsample, downsample_method, backend,
    )


def plot_khz_with_rms(
    hybrid_data: "HybridData",
    system: str,
    khz_variable: str,
    rms_variable: str | None = None,
    hours: list[int] | None = None,
    apply_calib: bool = True,
    rms_file_idx: int | None = None,
    rms_hours: list[int] | None = None,
    show: bool = True,
    save: str | None = None,
    backend: str | PlottingBackend = "matplotlib",
) -> tuple:
    """
    Plot kHz and RMS data together for comparison

    Parameters
    ----------
    hybrid_data : HybridData
        HybridData instance
    system : str
        FEPC system name
    khz_variable : str
        kHz variable name
    rms_variable : str, optional
        RMS variable name (defaults to khz_variable if None)
    hours : list of int, optional
        Hours to read for kHz data (also used for RMS if rms_hours is None)
    apply_calib : bool, optional
        Apply calibration to kHz data (default: True)
    rms_file_idx : int, optional
        Index of RMS file to load (ignored if rms_hours is provided)
    rms_hours : list of int, optional
        Hours to read for RMS data (defaults to hours if None)
    show : bool, optional
        Show plot (default: True)
    save : str, optional
        Save plot to file
    backend : str or PlottingBackend, optional
        Plotting backend (default: 'matplotlib').

    Returns
    -------
    tuple
        (fig, axes) — axes is a list of 2 for matplotlib, None otherwise.
    """
    logger.debug(
        "plot_khz_with_rms: system=%s, khz=%s, rms=%s",
        system, khz_variable, rms_variable,
    )

    if rms_variable is None:
        rms_variable = khz_variable
    if rms_hours is None and hours is not None:
        rms_hours = hours

    b = get_backend(backend)
    dfs: list[pd.DataFrame] = []
    fields: list[str] = []
    colors: list[str] = []
    field_styles: list[tuple] = []

    # kHz series (blue, thin solid line).
    try:
        khz_data, khz_time = hybrid_data.read_khz_variable(
            system, khz_variable, hours=hours, apply_calib=apply_calib
        )
        dfs.append(pd.DataFrame({"t": khz_time, khz_variable: khz_data}))
        fields.append(khz_variable)
        colors.append("blue")
        field_styles.append(("-", None, None, None))
    except (OSError, ValueError, RuntimeError, KeyError, struct.error) as e:
        logger.error("Error loading kHz variable %s: %s", khz_variable, e)

    # RMS series (red, dots).
    rms_col = f"{rms_variable} (RMS)"
    try:
        rms_data, rms_time = hybrid_data.read_rms_variable(
            system, rms_variable, file_idx=rms_file_idx, hours=rms_hours
        )
        dfs.append(pd.DataFrame({"t": rms_time, rms_col: rms_data}))
        fields.append(rms_col)
        colors.append("red")
        field_styles.append(("-", ".", None, None))
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        logger.error("Error loading RMS variable %s: %s", rms_variable, e)

    if not dfs:
        raise RuntimeError(f"No data loaded for {system}")

    merged = (
        pd.concat(dfs, ignore_index=True)
        .sort_values("t", kind="stable")
        .reset_index(drop=True)
    )

    title = f"{system} - {hybrid_data.date_str}"
    fig = plot_subplots(
        merged, fields, t_col="t",
        backend=b, title=title,
        colors=colors, field_styles=field_styles,
    )

    _handle_output(b, fig, show, save)

    axes = getattr(fig, "_magnetrun_axes", None)
    return fig, axes
