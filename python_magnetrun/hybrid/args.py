"""Argument parser definitions for the hybrid CLI."""

import argparse

from ..cli_args import args_to_outlier_config, create_outlier_parser  # noqa: F401
from ..data_dirs import HYBRID_DATA_DIR
from .hybrid_data import FEPC_SYSTEMS


def create_base_parser():
    """Create parser with base arguments.

    :return: ArgumentParser with base arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default=HYBRID_DATA_DIR or None,
        help=(
            "Base directory containing kHz, rms, trigger subdirectories "
            "(overrides MAGNETRUN_HYBRID_DATA_DIR / HYBRID_DATADIR)"
        ),
    )
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        help="Date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--fepc-system",
        "-s",
        type=str,
        choices=FEPC_SYSTEMS,
        help="FEPC system to use",
    )
    parser.add_argument(
        "--endian",
        "-e",
        type=str,
        choices=["big", "little"],
        default="big",
        help="Endianness of binary data (default: big)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Set logging level (default: WARNING)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (if not specified, logs to console)",
    )
    return parser


def create_info_parser():
    """Create parser with info/listing arguments.

    :return: ArgumentParser with info arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--list-dates",
        action="store_true",
        help="List available dates",
    )
    parser.add_argument(
        "--khz-vars",
        type=str,
        metavar="SYSTEM",
        help="Show kHz variables for a FEPC system",
    )
    parser.add_argument(
        "--rms-vars",
        type=str,
        metavar="SYSTEM",
        help="Show RMS variables for a FEPC system",
    )
    return parser


def create_plot_parser():
    """Create parser with plotting arguments.

    :return: ArgumentParser with plotting arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--plot-khz",
        type=str,
        metavar="VARIABLE",
        help="Plot kHz variable(s) (requires --fepc-system). "
        "Use comma-separated list for multiple variables (e.g., 'I,V,T')",
    )
    parser.add_argument(
        "--plot-rms",
        type=str,
        metavar="VARIABLE",
        help="Plot RMS variable(s) (requires --fepc-system). "
        "Use comma-separated list for multiple variables (e.g., 'I,V,T')",
    )
    parser.add_argument(
        "--plot-both",
        type=str,
        metavar="VARIABLE",
        help="Plot kHz and RMS data together (requires --fepc-system)",
    )
    parser.add_argument(
        "--rms-var",
        type=str,
        metavar="VARIABLE",
        help="RMS variable name for --plot-both (defaults to kHz variable name)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["subplots", "overlay"],
        default="subplots",
        help="Layout for multi-variable plots: 'subplots' (separate plots) or 'overlay' (same axes). Default: subplots",
    )
    parser.add_argument(
        "--hours",
        type=str,
        metavar="HOURS",
        help="Hours to plot for kHz data (comma-separated, e.g., '0,1,2')",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Do not apply calibration to kHz data",
    )
    parser.add_argument(
        "--save",
        type=str,
        metavar="FILE",
        help="Save plot to file",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the hybrid CLI.

    :return: Configured ArgumentParser with all arguments
    :rtype: argparse.ArgumentParser
    """
    base_parser = create_base_parser()
    info_parser = create_info_parser()
    plot_parser = create_plot_parser()
    outlier_parser = create_outlier_parser()

    parser = argparse.ArgumentParser(
        parents=[base_parser, info_parser, plot_parser, outlier_parser],
        description="Read hybrid magnet data (kHz, RMS, Trigger)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available dates
    python -m hybrid.cli --base-dir /data/hybrid --list-dates

    # Show summary for a specific date
    python -m hybrid.cli --base-dir /data/hybrid --date 2025-01-06

    # Show kHz variables
    python -m hybrid.cli --base-dir /data/hybrid --date 2025-01-06 --khz-vars FEPC-LNCMI

    # Show RMS variables
    python -m hybrid.cli --base-dir /data/hybrid --date 2025-01-06 --rms-vars FEPC-LNCMI

    # Plot a single kHz variable
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-khz ALIM1_J1

    # Plot multiple kHz variables (subplots)
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-khz ALIM1_J1,ALIM2_J1 --layout subplots

    # Plot multiple kHz variables (overlay on same axes)
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-khz ALIM1_J1,ALIM2_J1 --layout overlay

    # Plot a kHz variable for specific hours without calibration
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-khz ALIM1_J1 --hours 0,1,2 --no-calib

    # Plot a single RMS variable
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-rms ALIM1_J1

    # Plot multiple RMS variables
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-rms ALIM1_J1,ALIM2_J1 --layout overlay

    # Plot kHz and RMS together
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-both ALIM1_J1

    # Save plot to file
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-khz ALIM1_J1 --save output.png

    # Plot with outlier removal (IQR method)
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-khz ALIM1_J1 --remove-outliers iqr

    # Plot with outlier removal (zscore method, custom threshold)
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-khz ALIM1_J1 --remove-outliers zscore --outlier-threshold 3.0

    # Plot with rolling window outlier removal
    python -m hybrid.cli -d 2025-01-06 -s FEPC-AUX-LNCMI --plot-khz ALIM1_J1 --remove-outliers mad --outlier-window 1000
        """,
    )

    return parser
