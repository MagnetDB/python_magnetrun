"""
Plot trigger data from FEPC trigger files

This script reads and plots specific variables from trigger binary files.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python_magnetrun.hybrid.kHz.fepc_reader import CalibrationInfo
from python_magnetrun.hybrid.trigger.trigger_reader import (
    create_time_array,
    find_trigger_directories,
    parse_trigger_directory,
    read_trigger_data,
)
from python_magnetrun.log_utils import SIMPLE_FORMAT, setup_logging

logger = logging.getLogger(__name__)


def apply_calibration(
    data: np.ndarray, calib: CalibrationInfo, cnv_dir: Path | None = None
) -> np.ndarray:
    """
    Apply calibration to raw data

    Parameters:
    -----------
    data : np.ndarray
        Raw ADC data
    calib : CalibrationInfo
        Calibration parameters
    cnv_dir : Path, optional
        Directory containing CNV files

    Returns:
    --------
    np.ndarray : Calibrated data
    """
    if calib.cnv_file and cnv_dir:
        # Piecewise linear calibration
        cnv_path = cnv_dir / calib.cnv_file
        if cnv_path.exists():
            logger.info(f"Applying piecewise calibration from {calib.cnv_file}")
            # Load CNV lookup table
            try:
                cnv_data = np.loadtxt(cnv_path)
                # Interpolate
                calibrated = np.interp(data, cnv_data[:, 0], cnv_data[:, 1])
                return calibrated
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to apply CNV calibration: {e}")
                # Fall back to linear

    # Linear calibration: y = a * x + b
    return calib.a * data + calib.b


def plot_trigger_variable(
    trigger_dir: Path,
    system: str,
    variable: str,
    endian: str = "big",
    show_plot: bool = True,
    save_path: Path | None = None,
    apply_calib: bool = True,
    cnv_dir: Path | None = None,
    fig_size: tuple = (12, 6),
    title: str | None = None,
):
    """
    Plot a trigger variable

    Parameters:
    -----------
    trigger_dir : Path
        Path to trigger directory
    system : str
        'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
    variable : str
        Variable name to plot
    endian : str
        'big' or 'little'
    show_plot : bool
        Show interactive plot
    save_path : Path, optional
        Save plot to file
    apply_calib : bool
        Apply calibration
    cnv_dir : Path, optional
        Directory containing CNV files
    fig_size : tuple
        Figure size (width, height)
    title : str, optional
        Plot title
    """
    # Parse trigger info
    trigger_info = parse_trigger_directory(trigger_dir)

    # Read data
    logger.info(f"Reading {variable} from {trigger_dir.name}...")
    data, timestamp, config = read_trigger_data(
        trigger_dir, system, variable_name=variable, endian=endian
    )

    # Find card and channel info
    card = None
    channel = None
    for c in config.cards:
        if variable in c.variable_names:
            card = c
            channel = c.variable_names.index(variable)
            break

    if card is None:
        raise ValueError(f"Variable {variable} not found in configuration")

    assert (
        channel is not None
    ), f"Variable {variable} found in card but channel index is None"

    # Apply calibration if requested
    if apply_calib and card.calibrations and channel < len(card.calibrations):
        calib = card.calibrations[channel]

        if cnv_dir is None:
            # Try to find CNV directory in trigger system directory
            cnv_dir = trigger_dir / system

        logger.info(f"Applying calibration: a={calib.a}, b={calib.b}")
        data = apply_calibration(data, calib, cnv_dir)
        y_label = f"{variable} ({calib.unit})" if calib.unit else variable
    else:
        y_label = f"{variable} (raw ADC)"

    # Create time array
    time = create_time_array(len(data))

    # Convert to time relative to trigger (trigger at PRE seconds)
    pre_time = trigger_info.pre_samples / 10000.0  # 20s
    time_rel = time - pre_time

    # Create plot
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot data
    ax.plot(time_rel, data, linewidth=0.5, color="blue", alpha=0.7)

    # Mark trigger point
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Trigger", alpha=0.7)

    # Labels
    ax.set_xlabel("Time relative to trigger (s)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    if title is None:
        title = f"{variable} - {trigger_info.trigger_name}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add trigger info text
    info_text = (
        f"Trigger: {trigger_info.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
        f"Sample: {trigger_info.sample_idx}\n"
        f"PRE: {trigger_info.pre_samples / 10000:.1f}s, POST: {trigger_info.post_samples / 10000:.1f}s"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        logger.info(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show if requested
    if show_plot:
        plt.show()

    return fig, ax


def plot_multiple_triggers(
    base_dir: Path,
    date: str,
    system: str,
    variable: str,
    endian: str = "big",
    show_plot: bool = True,
    save_path: Path | None = None,
    apply_calib: bool = True,
    cnv_dir: Path | None = None,
    fig_size: tuple = (14, 8),
):
    """
    Plot same variable from multiple triggers on the same day

    Parameters:
    -----------
    base_dir : Path
        Base directory containing trigger subdirectory
    date : str
        Date in YYYY-MM-DD format
    system : str
        'FEPC-LNCMI' or 'FEPC-AUX-LNCMI'
    variable : str
        Variable name to plot
    endian : str
        'big' or 'little'
    show_plot : bool
        Show interactive plot
    save_path : Path, optional
        Save plot to file
    apply_calib : bool
        Apply calibration
    cnv_dir : Path, optional
        Directory containing CNV files
    fig_size : tuple
        Figure size (width, height)
    """
    # Find all triggers for this date
    triggers = find_trigger_directories(base_dir, date)

    if not triggers:
        raise ValueError(f"No triggers found for date {date}")

    logger.info(f"Found {len(triggers)} trigger(s) for {date}")

    # Create subplots
    n_triggers = len(triggers)
    fig, axes = plt.subplots(n_triggers, 1, figsize=fig_size, sharex=True)

    if n_triggers == 1:
        axes = [axes]

    # Plot each trigger
    for _i, (trigger_dir, ax) in enumerate(zip(triggers, axes, strict=False)):
        logger.info(f"Processing {trigger_dir.name}...")

        try:
            # Parse trigger info
            trigger_info = parse_trigger_directory(trigger_dir)

            # Read data
            data, timestamp, config = read_trigger_data(
                trigger_dir, system, variable_name=variable, endian=endian
            )

            # Find card and channel info
            card = None
            channel = None
            for c in config.cards:
                if variable in c.variable_names:
                    card = c
                    channel = c.variable_names.index(variable)
                    break

            # Apply calibration
            if (
                apply_calib
                and card
                and card.calibrations
                and channel is not None
                and channel < len(card.calibrations)
            ):
                calib = card.calibrations[channel]

                if cnv_dir is None:
                    cnv_dir = trigger_dir / system

                data = apply_calibration(data, calib, cnv_dir)
                y_label = f"{variable} ({calib.unit})" if calib.unit else variable
            else:
                y_label = f"{variable} (raw ADC)"

            # Create time array
            time = create_time_array(len(data))
            pre_time = trigger_info.pre_samples / 10000.0
            time_rel = time - pre_time

            # Plot
            ax.plot(time_rel, data, linewidth=0.5, color="blue", alpha=0.7)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.set_ylabel(y_label, fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add trigger time as title
            ax.set_title(
                trigger_info.timestamp.strftime("%H:%M:%S"), fontsize=10, loc="left"
            )

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error processing {trigger_dir.name}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

    # Common labels
    axes[-1].set_xlabel("Time relative to trigger (s)", fontsize=12)
    fig.suptitle(f"{variable} - {date} - All Triggers", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save if requested
    if save_path:
        logger.info(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    # Show if requested
    if show_plot:
        plt.show()

    return fig, axes


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Plot FEPC trigger data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single trigger
  python -m python_magnetrun.hybrid.trigger.plot_trigger_data --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \\
                               --system FEPC-LNCMI --variable I_H1

  # Plot all triggers for a date
  python -m python_magnetrun.hybrid.trigger.plot_trigger_data --base-dir /data/hybrid --date 2025-11-05 \\
                               --system FEPC-LNCMI --variable I_H1 --all

  # Save plot
  python -m python_magnetrun.hybrid.trigger.plot_trigger_data --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \\
                               --system FEPC-LNCMI --variable I_H1 --save trigger_plot.png

  # Skip calibration
  python -m python_magnetrun.hybrid.trigger.plot_trigger_data --trigger-dir /data/hybrid/trigger/TRIGGER_2025-11-05_08-16 \\
                               --system FEPC-LNCMI --variable I_H1 --no-calib
        """,
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory containing trigger subdirectory (for --all mode)",
    )

    parser.add_argument(
        "--trigger-dir", type=Path, help="Path to specific trigger directory"
    )

    parser.add_argument(
        "--date", type=str, help="Date in YYYY-MM-DD format (for --all mode)"
    )

    parser.add_argument(
        "--system",
        type=str,
        choices=["FEPC-LNCMI", "FEPC-AUX-LNCMI"],
        default="FEPC-LNCMI",
        help="FEPC system name",
    )

    parser.add_argument(
        "--variable", type=str, required=True, help="Variable name to plot"
    )

    parser.add_argument(
        "--all", action="store_true", help="Plot all triggers for specified date"
    )

    parser.add_argument(
        "--endian",
        type=str,
        choices=["big", "little"],
        default="big",
        help="Endianness of binary data",
    )

    parser.add_argument("--save", type=Path, help="Save plot to file")

    parser.add_argument(
        "--no-show", action="store_true", help="Do not show interactive plot"
    )

    parser.add_argument("--no-calib", action="store_true", help="Skip calibration")

    parser.add_argument(
        "--cnv-dir", type=Path, help="Directory containing CNV calibration files"
    )

    parser.add_argument(
        "--fig-size",
        type=str,
        default="12,6",
        help="Figure size as width,height (default: 12,6)",
    )

    args = parser.parse_args()
    setup_logging(fmt=SIMPLE_FORMAT)

    # Parse figure size
    try:
        fig_size = tuple(map(float, args.fig_size.split(",")))
    except ValueError:
        logger.warning("Invalid fig-size format, using default (12,6)")
        fig_size = (12, 6)

    try:
        if args.all:
            # Plot all triggers for date
            if not args.base_dir or not args.date:
                logger.error("Error: --base-dir and --date required for --all mode")
                return

            plot_multiple_triggers(
                args.base_dir,
                args.date,
                args.system,
                args.variable,
                args.endian,
                not args.no_show,
                args.save,
                not args.no_calib,
                args.cnv_dir,
                fig_size,
            )

        else:
            # Plot single trigger
            if not args.trigger_dir:
                logger.error("Error: --trigger-dir required")
                return

            plot_trigger_variable(
                args.trigger_dir,
                args.system,
                args.variable,
                args.endian,
                not args.no_show,
                args.save,
                not args.no_calib,
                args.cnv_dir,
                fig_size,
            )

    except (OSError, ValueError, RuntimeError) as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
