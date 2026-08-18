"""
FEPC Data Plotter - Read and plot data from binary files

This tool reads FEPC binary data files and plots specific variables.
Example: python -m python_magnetrun.hybrid.kHz.plot_fepc_data -c HOST_2_DATA.CFG -v ALIM1_J1 -s 4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python_magnetrun.hybrid.kHz.cfg_analyzer import (
    analyze_cfg_file,
    find_bin_files,
    get_variable_info,
)
from python_magnetrun.hybrid.kHz.fepc_reader import (
    apply_variable_calibration,
    read_variable_from_files,
)
from python_magnetrun.hybrid.utils import remove_outliers


def plot_variable(
    data: np.ndarray,
    time: np.ndarray,
    var_name: str,
    slot: int,
    output_file: str | None = None,
    unit: str | None = None,
    raw_data: np.ndarray | None = None,
    raw_time: np.ndarray | None = None,
    n_outliers: int = 0,
):
    """
    Plot variable data

    Parameters:
    -----------
    data : np.ndarray
        Variable data (cleaned if outliers removed)
    time : np.ndarray
        Time array in seconds
    var_name : str
        Variable name for title
    slot : int
        Slot number
    output_file : str, optional
        If provided, save plot to this file
    unit : str, optional
        Physical unit for y-axis label
    raw_data : np.ndarray, optional
        Original data before outlier removal (for comparison plot)
    raw_time : np.ndarray, optional
        Original time array
    n_outliers : int
        Number of outliers removed
    """
    # Use unit if provided, otherwise generic label
    ylabel = f"{var_name} ({unit})" if unit else f"{var_name} (Physical Units)"

    # If raw_data is provided, create comparison plot
    if raw_data is not None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Top plot: Original data with outliers
        axes[0].plot(raw_time, raw_data, linewidth=0.5, alpha=0.8, color="blue")
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(f"Original Data: {var_name} (Slot {slot})")
        axes[0].grid(True, alpha=0.3)
        axes[0].text(
            0.02,
            0.98,
            f"Min: {np.nanmin(raw_data):.2f}\nMax: {np.nanmax(raw_data):.2f}\nMean: {np.nanmean(raw_data):.2f}",
            transform=axes[0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Bottom plot: Cleaned data without outliers
        axes[1].plot(time, data, linewidth=0.5, alpha=0.8, color="green")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel(ylabel)
        pct = 100 * n_outliers / len(raw_data) if len(raw_data) > 0 else 0
        axes[1].set_title(
            f"Cleaned Data: {var_name} (Slot {slot}) - {n_outliers:,} outliers removed ({pct:.2f}%)"
        )
        axes[1].grid(True, alpha=0.3)
        axes[1].text(
            0.02,
            0.98,
            f"Min: {np.nanmin(data):.2f}\nMax: {np.nanmax(data):.2f}\nMean: {np.nanmean(data):.2f}",
            transform=axes[1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

        fig.tight_layout()

    else:
        # Single plot (no outlier removal)
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(time, data, linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Variable: {var_name} (Slot {slot})")
        ax.grid(True, alpha=0.3)

        # Add statistics
        ax.text(
            0.02,
            0.98,
            f"Min: {np.min(data):.2f}\nMax: {np.max(data):.2f}\nMean: {np.mean(data):.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150)
        print(f"\nPlot saved to: {output_file}")

    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Plot FEPC data from binary files")
    parser.add_argument(
        "-c",
        "--cfg",
        required=True,
        help="Path to HOST_X_DATA.CFG file",
    )
    parser.add_argument(
        "-v",
        "--variable",
        required=True,
        help="Variable name to plot (e.g., ALIM1_J1)",
    )
    parser.add_argument(
        "-s",
        "--slot",
        type=int,
        help="Card slot number (optional, will search all slots if not specified)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file for plot (PNG, PDF, etc.)",
    )
    parser.add_argument(
        "-d",
        "--date",
        nargs=2,
        help="Date range for files (YYYY-MM-DD start end)",
    )
    parser.add_argument(
        "-e",
        "--endian",
        choices=["big", "little"],
        default="big",
        help="Endianness of binary data: 'big' or 'little' (default: big)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: plot data incrementally as each file is loaded",
    )
    parser.add_argument(
        "--cnv-dir",
        default=".",
        help="Directory containing CNV calibration files (default: current directory)",
    )
    parser.add_argument(
        "--remove-outliers",
        choices=["iqr", "zscore", "mad", "percentile"],
        help="Remove outliers using specified method",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=1.5,
        help="Threshold for outlier detection (default: 1.5 for IQR, 3.0 for zscore/mad)",
    )
    parser.add_argument(
        "--outlier-window",
        type=int,
        help="Window size for rolling outlier detection (optional)",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Loading configuration from: {args.cfg}")
        config = analyze_cfg_file(args.cfg, verbose=False)

        # Find variable
        print(f"\nSearching for variable: {args.variable}")
        var_info = get_variable_info(config, args.variable)

        if not var_info:
            print(f"❌ Variable '{args.variable}' not found in configuration")
            print("\nAvailable variables by slot:")
            for card in config.cards:
                print(
                    f"  Slot {card.slot} ({card.card_type}): {card.variable_names[:3]}..."
                )
            return None

        slot = var_info["slot"]
        channel_idx = var_info["channel_idx"]
        card_type = var_info["card_type"]

        print(f"✓ Found in slot {slot}, channel {channel_idx} ({card_type} card)")

        # Get unit from calibration if available
        unit = None
        calibration = var_info.get("calibration")
        if calibration and hasattr(calibration, "unit"):
            unit = calibration.unit
            if unit:
                print(f"  Unit: {unit}")

        # Find bin files
        print("\nSearching for bin files...")
        bin_files = find_bin_files(args.cfg, slot, args.date)

        if not bin_files:
            print(f"❌ No bin files found for slot {slot}")
            return None

        print(f"✓ Found {len(bin_files)} file(s)")

        # Read data
        print("\nReading data...")
        data, time = read_variable_from_files(
            bin_files,
            slot,
            channel_idx,
            card_type,
            config,
            endian=args.endian,
            debug=args.debug,
            var_name=args.variable,
        )

        # Apply calibration
        print("Applying calibration...")
        cfg_dir = Path(args.cfg).parent
        cnv_dir = args.cnv_dir if args.cnv_dir != "." else str(cfg_dir)
        data = apply_variable_calibration(
            data, config, slot, channel_idx, cnv_directory=cnv_dir
        )

        print(f"✓ Loaded {len(data):,} samples over {time[-1]:.1f} seconds")

        # Remove outliers if requested
        raw_data, raw_time, n_outliers = None, None, 0
        if args.remove_outliers:
            # Keep original data for comparison plot
            raw_data = data.copy()
            raw_time = time.copy()

            print(f"\nRemoving outliers using {args.remove_outliers} method...")
            data, time, n_outliers = remove_outliers(
                data,
                time,
                method=args.remove_outliers,
                threshold=args.outlier_threshold,
                window_size=args.outlier_window,
            )
            pct = 100 * n_outliers / len(raw_data)
            print(f"✓ Removed {n_outliers:,} outliers ({pct:.2f}%)")

        # Plot
        print("\nPlotting...")
        plot_variable(
            data,
            time,
            args.variable,
            slot,
            args.output,
            unit=unit,
            raw_data=raw_data,
            raw_time=raw_time,
            n_outliers=n_outliers,
        )

        return config

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    except (OSError, ValueError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    config = main()
