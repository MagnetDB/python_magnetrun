"""
FEPC Data Plotter - Read and plot data from binary files

This tool reads FEPC binary data files and plots specific variables.
Example: python -m python_magnetrun.hybrid.kHz.plot_fepc_data -c HOST_2_DATA.CFG -v ALIM1_J1 -s 4
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python_magnetrun.hybrid.kHz.cfg_analyzer import analyze_cfg_file, extract_host_number
from python_magnetrun.hybrid.kHz.fepc_reader import (
    KHZ_SAMPLING_FREQUENCY,
    apply_calibration,
    read_hour_file,
)
from python_magnetrun.hybrid.utils import remove_outliers


def find_bin_files(cfg_path: str, slot: int, date_range: tuple[str, str] | None = None) -> list:
    """
    Find all bin files matching the pattern for a given slot

    Parameters:
    -----------
    cfg_path : str
        Path to CFG file (used to extract HOST number)
    slot : int
        Card slot number
    date_range : tuple of str, optional
        Date range filter (YYYY-MM-DD format)
        If None, finds all files for today

    Returns:
    --------
    list
        List of paths to matching bin files, sorted by hour
    """
    cfg_dir = Path(cfg_path).parent
    host_number = extract_host_number(cfg_path)

    # Build file pattern: XXHOST_n_LIST_{slot}.bin
    pattern = f"*HOST_{host_number}_LIST_{slot}.bin"

    matching_files = sorted(cfg_dir.glob(pattern))

    if not matching_files:
        print(f"No bin files found matching pattern: {pattern}")
        return []

    # Filter by date range if provided
    if date_range:
        start_date, end_date = date_range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        filtered = []
        for f in matching_files:
            # Extract hour from filename (XX in XXHOST_...)
            match = re.match(r"(\d{2})HOST", f.name)
            if match:
                hour = int(match.group(1))
                # This is simplified; actual date info would need more context
                filtered.append(f)
        matching_files = filtered

    return matching_files


def get_variable_info(config, var_name: str) -> dict | None:
    """
    Find variable in config and return its slot and channel index

    Parameters:
    -----------
    config : FEPCConfig
        Configuration object
    var_name : str
        Variable name to search for

    Returns:
    --------
    dict or None
        Dictionary with 'slot', 'channel_idx', 'card_type', 'card', 'calibration'
        or None if not found
    """
    var_upper = var_name.upper()

    for card in config.cards:
        for ch_idx, var in enumerate(card.variable_names):
            if var.upper() == var_upper:
                # Get calibration if available
                calib = None
                if card.calibrations and ch_idx < len(card.calibrations):
                    calib = card.calibrations[ch_idx]
                return {
                    "slot": card.slot,
                    "channel_idx": ch_idx,
                    "card_type": card.card_type,
                    "card": card,
                    "calibration": calib,
                }

    return None


def read_variable_from_files(
    bin_files: list,
    slot: int,
    channel_idx: int,
    card_type: str,
    config=None,
    endian: str = "big",
    debug: bool = False,
    var_name: str = "Variable",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read specific variable from all bin files

    Parameters:
    -----------
    bin_files : list
        List of bin file paths
    slot : int
        Card slot number
    channel_idx : int
        Channel index within the card
    card_type : str
        'ANA' for analog or 'DIG' for digital
    config : FEPCConfig, optional
        Configuration object for calibration
    endian : str
        Endianness: 'big' or 'little' (default: 'big')
    debug : bool
        If True, plot data after each block is loaded (default: False)
    var_name : str
        Variable name for debug plot title

    Returns:
    --------
    data : np.ndarray
        Concatenated data from all files
    time : np.ndarray
        Time array in seconds
    """
    all_data = []
    total_samples = 0

    # Get sampling frequency
    sampling_freq = KHZ_SAMPLING_FREQUENCY

    # Setup debug plot if requested
    fig, ax, line = None, None, None
    if debug:
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(f"{var_name}")
        ax.set_title(f"Debug: {var_name} (Slot {slot}) - Loading...")
        ax.grid(True, alpha=0.3)
        (line,) = ax.plot([], [], linewidth=0.5, alpha=0.8)
        plt.show(block=False)

    if debug:
        print(f"Reading {len(bin_files)} files... (endian={endian})", flush=True)

    for file_idx, file_path in enumerate(bin_files):
        try:
            # Read complete hour file
            if debug:
                print(f"Reading file: {file_path.name}", flush=True)
            hour_data = read_hour_file(
                str(file_path),
                card_type,
                endian=endian,
                debug=debug,
                debug_channel=channel_idx,
            )
            all_data.append(hour_data[:, channel_idx])
            total_samples += len(hour_data)
            print(f"  ✓ {file_path.name}")

            # Update debug plot
            if debug:
                current_data = np.concatenate(all_data)
                current_time = np.arange(len(current_data)) / sampling_freq
                if line is not None:
                    line.set_data(current_time, current_data)
                if ax is not None:
                    ax.relim()
                    ax.autoscale_view()
                    ax.set_title(
                        f"Debug: {var_name} (Slot {slot}) - File {file_idx + 1}/{len(bin_files)}"
                    )
                if fig is not None:
                    fig.canvas.draw()
                if fig is not None:
                    fig.canvas.flush_events()
                plt.pause(1)

        except (OSError, ValueError, RuntimeError) as e:
            print(f"  ✗ Error reading {file_path.name}: {e}")

    if debug:
        plt.ioff()  # Turn off interactive mode
        plt.close(fig)

    if not all_data:
        raise ValueError("No data could be read from bin files")

    # Concatenate all data
    data = np.concatenate(all_data)

    # Create time array
    time = np.arange(len(data)) / sampling_freq

    return data, time


def apply_variable_calibration(
    data: np.ndarray, config, slot: int, channel_idx: int, cnv_directory: str = "."
) -> np.ndarray:
    """
    Apply calibration to variable data

    Parameters:
    -----------
    data : np.ndarray
        Raw data
    config : FEPCConfig
        Configuration object
    slot : int
        Card slot
    channel_idx : int
        Channel index
    cnv_directory : str
        Directory containing CNV files for piecewise calibration

    Returns:
    --------
    np.ndarray
        Calibrated data
    """
    from pathlib import Path

    from .fepc_reader import load_calibration

    try:
        card = config.get_card_by_slot(slot)
        if card.card_type != "ANA":
            # Digital data doesn't need calibration
            return data.astype(np.float64)

        if card.calibrations and channel_idx < len(card.calibrations):
            calib_info = card.calibrations[channel_idx]

            # Check if CNV file calibration should be used
            if calib_info.cnv_file:
                cnv_path = Path(cnv_directory) / calib_info.cnv_file
                if cnv_path.exists():
                    print(f"  Using CNV file calibration: {calib_info.cnv_file}")
                    cnv_dict = load_calibration(str(cnv_path))
                    return apply_calibration(data, cnv_dict=cnv_dict)
                else:
                    print(f"  Warning: CNV file {cnv_path} not found, using linear calibration")

            # Use linear calibration
            print(
                f"  Using linear calibration: A={calib_info.a}, B={calib_info.b}, COEFA={calib_info.coef_a}, COEFB={calib_info.coef_b}"
            )
            return apply_calibration(data, calib_info)

    except (ValueError, IndexError, AttributeError) as e:
        print(f"  Warning: Could not apply calibration: {e}")

    return data.astype(np.float64)


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

        plt.tight_layout()

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

        plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
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
                print(f"  Slot {card.slot} ({card.card_type}): {card.variable_names[:3]}...")
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
        data = apply_variable_calibration(data, config, slot, channel_idx, cnv_directory=cnv_dir)

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
