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
- Outlier detection is now handled by the separate `python_magnetrun.hybrid.outliers` module
- For outlier analysis, use `from python_magnetrun.hybrid.outliers import OutlierDetector, detect_outliers`
- Plotting functions accept pre-computed outlier masks via `outlier_mask` parameter
"""

import logging
import struct
from typing import TYPE_CHECKING, Optional

import numpy as np

from python_magnetrun.utils.downsampling import DownsampleConfig, downsample_arrays

# Type checking import to avoid circular import
if TYPE_CHECKING:
    from .hybrid_data import HybridData
    from .outliers import OutlierResult

# Import outlier detection from dedicated module
from .outliers import OutlierResult

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
        Dictionary mapping variable names to OutlierResult objects
        from pre-computed outlier detection. Use hybrid.outliers.detect_outliers()
        or OutlierDetector to compute outliers before plotting.
    outlier_strategy : str, optional
        Strategy for handling outliers: 'remove', 'interpolate', 'highlight', 'none'
        Default: 'interpolate'
    downsample : int, optional
        Target number of points for plotting (default: 50000).
        Set to None to disable downsampling.
        For 1-hour kHz data (~3.6M points), 50000 provides good balance.
    downsample_method : str, optional
        Downsampling method: 'auto', 'minmax_lttb', 'stride', 'minmax'
    **plot_kwargs : dict
        Additional arguments passed to plt.plot()

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes (array for subplots, single for overlay)

    Example
    -------
    >>> from python_magnetrun.hybrid.outliers import detect_outliers
    >>> # Pre-compute outliers for each variable
    >>> outlier_results = {}
    >>> for var in ['U1', 'U2', 'I1']:
    ...     data, time = hybrid_data.read_khz_variable(system, var)
    ...     outlier_results[var] = detect_outliers(data, method='iqr')
    >>> # Plot with pre-computed outliers
    >>> fig, axes = plot_khz_variables(
    ...     hybrid_data, system, ['U1', 'U2', 'I1'],
    ...     outlier_results=outlier_results,
    ...     outlier_strategy='highlight'
    ... )
    """
    import matplotlib.pyplot as plt

    n_vars = len(variables)
    if n_vars == 0:
        raise ValueError("At least one variable must be specified")

    # Single variable - delegate to single variable function
    if n_vars == 1:
        var_outlier = outlier_results.get(variables[0]) if outlier_results else None
        return plot_khz_variable(
            hybrid_data,
            system,
            variables[0],
            hours=hours,
            apply_calib=apply_calib,
            cnv_dir=cnv_dir,
            show=show,
            save=save,
            outlier_result=var_outlier,
            outlier_strategy=outlier_strategy,
            downsample=downsample,
            downsample_method=downsample_method,
            **plot_kwargs,
        )

    logger.debug(f"plot_khz_variables: system={system}, variables={variables}")

    # Create figure based on layout
    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(14, 8))
        axes = [ax]
        colors = plt.cm.tab10.colors
    else:  # subplots
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars), sharex=share_x)
        if n_vars == 1:
            axes = [axes]
        colors = None

    # Plot each variable
    for i, variable in enumerate(variables):
        try:
            data, time = hybrid_data.read_khz_variable(
                system, variable, hours=hours, apply_calib=apply_calib, cnv_dir=cnv_dir
            )

            # Get outlier result for this variable if provided
            var_outlier = outlier_results.get(variable) if outlier_results else None
            highlight_outliers = False

            # Apply outlier handling if result provided
            if var_outlier is not None and outlier_strategy != "none":
                if outlier_strategy == "highlight":
                    highlight_outliers = True
                else:
                    # Apply strategy to data
                    data = var_outlier.apply_to_data(data, strategy=outlier_strategy)
                    # Remove NaN values if strategy was 'remove' or 'nan'
                    if outlier_strategy in ("remove", "nan"):
                        valid_mask = ~np.isnan(data)
                        data = data[valid_mask]
                        time = time[valid_mask]
                    logger.info(
                        f"Applied {outlier_strategy} to {var_outlier.n_outliers} "
                        f"outliers in {variable}"
                    )

            # Apply downsampling for efficient plotting
            if downsample is not None and len(data) > downsample:
                original_len = len(data)
                data, time = downsample_arrays(
                    data, time, _make_downsample_config(downsample, downsample_method)
                )
                logger.info(
                    f"Downsampled {variable}: {original_len:,} -> {len(data):,} points"
                )

            unit = _get_khz_unit(hybrid_data, system, variable)
            ylabel = f"{variable}" + (f" [{unit}]" if unit else "")

            if layout == "overlay":
                ax = axes[0]
                color = colors[i % len(colors)]
                ax.plot(time, data, label=variable, color=color, **plot_kwargs)
                # Highlight outliers if requested (on overlay)
                if highlight_outliers and var_outlier is not None:
                    outlier_data = data[var_outlier.mask[: len(data)]]
                    outlier_time = time[var_outlier.mask[: len(time)]]
                    if len(outlier_data) > 0:
                        ax.scatter(
                            outlier_time,
                            outlier_data,
                            c="red",
                            s=10,
                            alpha=0.7,
                            label=f"{variable} outliers",
                            zorder=5,
                        )
            else:
                ax = axes[i]
                ax.plot(time, data, label=variable, **plot_kwargs)
                # Highlight outliers if requested
                if highlight_outliers and var_outlier is not None:
                    outlier_data = data[var_outlier.mask[: len(data)]]
                    outlier_time = time[var_outlier.mask[: len(time)]]
                    if len(outlier_data) > 0:
                        ax.scatter(
                            outlier_time,
                            outlier_data,
                            c="red",
                            s=10,
                            alpha=0.7,
                            label="Outliers",
                            zorder=5,
                        )
                ax.set_ylabel(ylabel)
                ax.set_title(f"{variable}")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")

        except (OSError, ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Error plotting {variable}: {e}")
            if layout != "overlay":
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error: {e}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{variable} (ERROR)")

    # Finalize plot
    if layout == "overlay":
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Value")
        axes[0].set_title(f"{system} - kHz Data ({hybrid_data.date_str})")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"{system} - kHz Data ({hybrid_data.date_str})",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, axes if layout != "overlay" else axes[0]


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
        Dictionary mapping variable names to OutlierResult objects
        from pre-computed outlier detection. Use hybrid.outliers.detect_outliers()
        or OutlierDetector to compute outliers before plotting.
    outlier_strategy : str, optional
        Strategy for handling outliers: 'remove', 'interpolate', 'highlight', 'none'
        Default: 'interpolate'
    **plot_kwargs : dict
        Additional arguments passed to plt.plot()

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    n_vars = len(variables)
    if n_vars == 0:
        raise ValueError("At least one variable must be specified")

    # Single variable - delegate to single variable function
    if n_vars == 1:
        var_outlier = outlier_results.get(variables[0]) if outlier_results else None
        return plot_rms_variable(
            hybrid_data,
            system,
            variables[0],
            file_idx=file_idx,
            hours=hours,
            show=show,
            save=save,
            outlier_result=var_outlier,
            outlier_strategy=outlier_strategy,
            **plot_kwargs,
        )

    logger.debug(f"plot_rms_variables: system={system}, variables={variables}")

    # Create figure based on layout
    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(14, 8))
        axes = [ax]
        colors = plt.cm.tab10.colors
    else:  # subplots
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars), sharex=share_x)
        if n_vars == 1:
            axes = [axes]
        colors = None

    # Plot each variable
    for i, variable in enumerate(variables):
        try:
            data, time = hybrid_data.read_rms_variable(
                system, variable, file_idx=file_idx, hours=hours
            )

            # Get outlier result for this variable if provided
            var_outlier = outlier_results.get(variable) if outlier_results else None
            highlight_outliers = False

            # Apply outlier handling if result provided
            if var_outlier is not None and outlier_strategy != "none":
                if outlier_strategy == "highlight":
                    highlight_outliers = True
                else:
                    # Apply strategy to data
                    data = var_outlier.apply_to_data(data, strategy=outlier_strategy)
                    # Remove NaN values if strategy was 'remove' or 'nan'
                    if outlier_strategy in ("remove", "nan"):
                        valid_mask = ~np.isnan(data)
                        data = data[valid_mask]
                        time = time[valid_mask]
                    logger.info(
                        f"Applied {outlier_strategy} to {var_outlier.n_outliers} "
                        f"outliers in {variable}"
                    )

            info_idx = file_idx if file_idx is not None else 0
            unit = _get_rms_unit(hybrid_data, system, variable, info_idx)
            ylabel = f"{variable}" + (f" [{unit}]" if unit else "")

            if layout == "overlay":
                ax = axes[0]
                color = colors[i % len(colors)]
                ax.plot(time, data, label=variable, color=color, **plot_kwargs)
                # Highlight outliers if requested
                if highlight_outliers and var_outlier is not None:
                    outlier_data = data[var_outlier.mask[: len(data)]]
                    outlier_time = time[var_outlier.mask[: len(time)]]
                    if len(outlier_data) > 0:
                        ax.scatter(
                            outlier_time,
                            outlier_data,
                            c="red",
                            s=10,
                            alpha=0.7,
                            label=f"{variable} outliers",
                            zorder=5,
                        )
            else:
                ax = axes[i]
                ax.plot(time, data, label=variable, **plot_kwargs)
                # Highlight outliers if requested
                if highlight_outliers and var_outlier is not None:
                    outlier_data = data[var_outlier.mask[: len(data)]]
                    outlier_time = time[var_outlier.mask[: len(time)]]
                    if len(outlier_data) > 0:
                        ax.scatter(
                            outlier_time,
                            outlier_data,
                            c="red",
                            s=10,
                            alpha=0.7,
                            label="Outliers",
                            zorder=5,
                        )
                ax.set_ylabel(ylabel)
                ax.set_title(f"{variable}")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")

        except (OSError, ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Error plotting {variable}: {e}")
            if layout != "overlay":
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error: {e}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{variable} (ERROR)")

    # Finalize plot
    if layout == "overlay":
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Value")
        axes[0].set_title(f"{system} - RMS Data ({hybrid_data.date_str})")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"{system} - RMS Data ({hybrid_data.date_str})",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, axes if layout != "overlay" else axes[0]


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
        Axes to plot on (creates new figure if None)
    show : bool, optional
        Show plot (default: True)
    save : str, optional
        Save plot to file
    outlier_result : OutlierResult, optional
        Pre-computed outlier detection result from python_magnetrun.hybrid.outliers module.
        If provided, outliers will be handled according to outlier_strategy.
        Use `from python_magnetrun.hybrid.outliers import OutlierDetector` to detect outliers separately.
    outlier_strategy : str, optional
        How to handle outliers when outlier_result is provided:
        - 'remove': Remove outlier points
        - 'interpolate': Replace with interpolated values (default)
        - 'highlight': Plot outliers in different color
        - 'none': Ignore outlier_result
    downsample : int, optional
        Target number of points for plotting (default: 50000).
        Set to None to disable downsampling.
    downsample_method : str, optional
        Downsampling method: 'auto', 'minmax_lttb', 'stride', 'minmax'
    **plot_kwargs : dict
        Additional arguments passed to plt.plot()

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes

    Examples
    --------
    >>> # Simple plot (no outlier handling)
    >>> fig, ax = plot_khz_variable(data, "FEPC-LNCMI", "I_H1")

    >>> # With pre-computed outlier detection
    >>> from python_magnetrun.hybrid.outliers import OutlierDetector
    >>> detector = OutlierDetector(method='iqr', threshold=1.5)
    >>> result = detector.detect(raw_data)
    >>> fig, ax = plot_khz_variable(data, "FEPC-LNCMI", "I_H1",
    ...                              outlier_result=result, outlier_strategy='highlight')
    """
    import matplotlib.pyplot as plt

    logger.debug(f"plot_khz_variable: system={system}, variable={variable}")

    # Read data
    data, time = hybrid_data.read_khz_variable(
        system, variable, hours=hours, apply_calib=apply_calib, cnv_dir=cnv_dir
    )

    # Get unit if available
    config = hybrid_data.load_khz_config(system)
    if config is None:
        raise ValueError(f"No configuration found for {system}")

    unit = ""
    if config:
        for card in config.cards:
            if variable in card.variable_names:
                idx = card.variable_names.index(variable)
                if (
                    card.calibrations
                    and idx < len(card.calibrations)
                    and card.calibrations[idx]
                ):
                    unit = card.calibrations[idx].unit or ""
                break

    ylabel = f"{variable}"
    if unit:
        ylabel += f" [{unit}]"

    # Handle outliers if result provided
    plot_data, plot_time = data, time
    outlier_info = ""

    if outlier_result is not None and outlier_strategy != "none":
        if outlier_strategy == "highlight":
            # Will plot outliers separately in different color
            pass  # Handle below
        elif outlier_strategy in ["remove", "interpolate", "nan", "clip", "median"]:
            plot_data, plot_time = outlier_result.apply_to_data(
                data, time, strategy=outlier_strategy
            )
            outlier_info = (
                f" ({outlier_result.n_outliers:,} outliers {outlier_strategy}d)"
            )
            logger.info(
                f"Applied outlier handling: {outlier_result.n_outliers:,} outliers "
                f"({outlier_result.outlier_percentage:.2f}%) using {outlier_strategy}"
            )

    # Apply downsampling for efficient plotting
    if downsample is not None and len(plot_data) > downsample:
        original_len = len(plot_data)
        plot_data, plot_time = downsample_arrays(
            plot_data, plot_time, _make_downsample_config(downsample, downsample_method)
        )
        logger.info(
            f"Downsampled {variable}: {original_len:,} -> {len(plot_data):,} points"
        )

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    # Plot main data
    label = plot_kwargs.pop("label", f"{variable}{outlier_info}")
    ax.plot(plot_time, plot_data, label=label, **plot_kwargs)

    # Highlight outliers if requested
    if outlier_result is not None and outlier_strategy == "highlight":
        outlier_mask = outlier_result.mask
        if downsample is not None and len(data) > downsample:
            # Need to downsample outlier data too
            outlier_data = data[outlier_mask]
            outlier_time = time[outlier_mask]
            if len(outlier_data) > downsample // 10:  # Use fewer points for outliers
                outlier_data, outlier_time = downsample_arrays(
                    outlier_data,
                    outlier_time,
                    DownsampleConfig(n_out=downsample // 10, method="stride"),
                )
        else:
            outlier_data = data[outlier_mask]
            outlier_time = time[outlier_mask]

        ax.scatter(
            outlier_time,
            outlier_data,
            c="red",
            s=10,
            alpha=0.5,
            label=f"Outliers ({outlier_result.n_outliers:,})",
            zorder=5,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{system} - {variable} ({hybrid_data.date_str})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, ax


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
        Index of RMS file to load (used if hours is None, default: 0)
    hours : list of int, optional
        List of hours to load (0-23). If provided, file_idx is ignored.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new figure if None)
    show : bool, optional
        Show plot (default: True)
    save : str, optional
        Save plot to file
    outlier_result : OutlierResult, optional
        Pre-computed outlier detection result from python_magnetrun.hybrid.outliers module.
        If provided, outliers will be handled according to outlier_strategy.
        Use `from python_magnetrun.hybrid.outliers import OutlierDetector` to detect outliers separately.
    outlier_strategy : str, optional
        How to handle outliers when outlier_result is provided:
        - 'remove': Remove outlier points
        - 'interpolate': Replace outliers with interpolated values
        - 'highlight': Plot outliers in red on top of data
        - 'none': Ignore outlier_result and plot raw data
        Default: 'interpolate'
    downsample : int, optional
        Target number of points for plotting.
        Set to None to disable downsampling (RMS data is usually smaller).
    downsample_method : str, optional
        Downsampling method: 'auto', 'minmax_lttb', 'stride', 'minmax'
    **plot_kwargs : dict
        Additional arguments passed to plt.plot()

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes

    Example
    -------
    >>> from python_magnetrun.hybrid.outliers import detect_outliers
    >>> # Read data and detect outliers separately
    >>> data, time = hybrid_data.read_rms_variable(system, 'U1')
    >>> outlier_result = detect_outliers(data, method='iqr', threshold=1.5)
    >>> # Plot with pre-computed outliers
    >>> fig, ax = plot_rms_variable(
    ...     hybrid_data, system, 'U1',
    ...     outlier_result=outlier_result,
    ...     outlier_strategy='highlight'
    ... )
    """
    import matplotlib.pyplot as plt

    logger.debug(
        f"plot_rms_variable: system={system}, variable={variable}, hours={hours}"
    )

    # Read data using read_rms_variable (same pattern as kHz)
    data, time = hybrid_data.read_rms_variable(
        system, variable, file_idx=file_idx, hours=hours
    )

    # Get unit if available from variable info
    unit = ""
    try:
        # Use file_idx=0 for variable info if hours specified
        info_idx = file_idx if file_idx is not None else 0
        var_info = hybrid_data.get_rms_variable_info(system, file_idx=info_idx)
        var_row = var_info[var_info["name"] == variable]
        if not var_row.empty and var_row.iloc[0]["unit"]:
            unit = var_row.iloc[0]["unit"]
    except (OSError, ValueError, RuntimeError, KeyError):
        pass  # No unit available

    ylabel = f"{variable}"
    if unit:
        ylabel += f" [{unit}]"

    # Handle outliers if result provided
    highlight_outliers = False
    if outlier_result is not None and outlier_strategy != "none":
        if outlier_strategy == "highlight":
            highlight_outliers = True
        else:
            # Apply strategy to data
            data = outlier_result.apply_to_data(data, strategy=outlier_strategy)
            # Remove NaN values if strategy was 'remove' or 'nan'
            if outlier_strategy in ("remove", "nan"):
                valid_mask = ~np.isnan(data)
                data = data[valid_mask]
                time = time[valid_mask]
            logger.info(
                f"Applied {outlier_strategy} to {outlier_result.n_outliers} outliers in {variable}"
            )

    # Apply downsampling if requested
    if downsample is not None and len(data) > downsample:
        original_len = len(data)
        data, time = downsample_arrays(
            data, time, _make_downsample_config(downsample, downsample_method)
        )
        logger.info(f"Downsampled {variable}: {original_len:,} -> {len(data):,} points")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    # Plot main data
    label = plot_kwargs.pop("label", f"{variable} (RMS)")
    ax.plot(time, data, label=label, **plot_kwargs)

    # Highlight outliers if requested
    if highlight_outliers and outlier_result is not None:
        # Need to re-read original data for highlighting
        orig_data, orig_time = hybrid_data.read_rms_variable(
            system, variable, file_idx=file_idx, hours=hours
        )
        outlier_mask = outlier_result.mask[: len(orig_data)]
        outlier_data = orig_data[outlier_mask]
        outlier_time = orig_time[outlier_mask]
        if len(outlier_data) > 0:
            ax.scatter(
                outlier_time,
                outlier_data,
                c="red",
                s=20,
                alpha=0.7,
                label=f"Outliers ({outlier_result.n_outliers})",
                zorder=5,
            )
            logger.info(
                f"Highlighting {outlier_result.n_outliers} outliers in {variable}"
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    title = f"{system} RMS - {variable} ({hybrid_data.date_str})"
    if outlier_result is not None and outlier_strategy != "none":
        title += f"\n[{outlier_strategy}: {outlier_result.n_outliers} outliers]"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, ax


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

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes array
    """
    import matplotlib.pyplot as plt

    logger.debug(
        f"plot_khz_with_rms: system={system}, khz={khz_variable}, rms={rms_variable}"
    )

    if rms_variable is None:
        rms_variable = khz_variable

    # Use same hours for RMS if rms_hours not specified
    if rms_hours is None and hours is not None:
        rms_hours = hours

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot kHz data
    try:
        khz_data, khz_time = hybrid_data.read_khz_variable(
            system, khz_variable, hours=hours, apply_calib=apply_calib
        )
        axes[0].plot(
            khz_time, khz_data, "b-", linewidth=0.5, label=f"{khz_variable} (kHz)"
        )
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel(khz_variable)
        axes[0].set_title(f"kHz Data: {khz_variable}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    except ValueError as e:
        logger.error(f"Value error plotting kHz variable: {e}")
        axes[0].text(
            0.5,
            0.5,
            f"Error loading kHz data:\n{e}",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title(f"kHz Data: {khz_variable} (ERROR)")
    except struct.error as e:
        logger.critical(f"Struct error plotting kHz variable: {e}")
    except (OSError, RuntimeError, KeyError) as e:
        logger.error(f"Error plotting kHz variable: {e}")
        axes[0].text(
            0.5,
            0.5,
            f"Error loading kHz data:\n{e}",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title(f"kHz Data: {khz_variable} (ERROR)")

    # Plot RMS data (using read_rms_variable for consistency)
    try:
        rms_data, rms_time = hybrid_data.read_rms_variable(
            system, rms_variable, file_idx=rms_file_idx, hours=rms_hours
        )
        axes[1].plot(
            rms_time,
            rms_data,
            "r-",
            marker=".",
            markersize=2,
            label=f"{rms_variable} (RMS)",
        )
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel(rms_variable)
        axes[1].set_title(f"RMS Data: {rms_variable}")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    except ValueError as e:
        logger.error(f"Value error plotting RMS variable: {e}")
        axes[1].text(
            0.5,
            0.5,
            f"Error loading RMS data:\n{e}",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title(f"RMS Data: {rms_variable} (ERROR)")
    except (OSError, RuntimeError, KeyError) as e:
        logger.error(f"Error plotting RMS variable: {e}")
        axes[1].text(
            0.5,
            0.5,
            f"Error loading RMS data:\n{e}",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title(f"RMS Data: {rms_variable} (ERROR)")

    fig.suptitle(f"{system} - {hybrid_data.date_str}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save}")

    if show:
        plt.show()

    return fig, axes
