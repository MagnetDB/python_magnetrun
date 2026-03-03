"""
Command-line interface for hybrid magnet data (kHz, RMS, Trigger)

This module provides CLI tools for reading and plotting hybrid magnet data
from FEPC acquisition systems.

Usage:
    python -m python_magnetrun.hybrid.cli --help
    python -m python_magnetrun.hybrid.cli --base-dir /data/hybrid --date 2025-01-06
"""

import argparse
import logging

from .hybrid_data import HybridData, FEPC_SYSTEMS
from .utils import list_available_dates, log_exception, format_exception_location

# Setup logger
logger = logging.getLogger(__name__)


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
        default=None,
        help="Base directory containing kHz, rms, trigger subdirectories",
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


def create_outlier_parser():
    """Create parser with outlier removal arguments.

    :return: ArgumentParser with outlier arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--remove-outliers",
        type=str,
        metavar="METHOD",
        choices=["iqr", "zscore", "mad", "percentile"],
        help="Remove outliers using specified method (iqr, zscore, mad, percentile)",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=1.5,
        metavar="THRESHOLD",
        help="Threshold for outlier detection (default: 1.5 for IQR, 3.0 for zscore)",
    )
    parser.add_argument(
        "--outlier-window",
        type=int,
        metavar="SIZE",
        help="Rolling window size for local outlier detection (optional)",
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


def run_list_dates(args) -> None:
    """Handle --list-dates command."""
    print("Available dates:")
    for data_type in ["kHz", "rms", "trigger"]:
        dates = list_available_dates(args.base_dir, data_type)
        if dates:
            print(
                f"  {data_type}: {', '.join(dates[:5])}"
                + (f" ... ({len(dates)} total)" if len(dates) > 5 else "")
            )


def run_show_khz_vars(data: HybridData, system: str) -> None:
    """Handle --khz-vars command."""
    print(f"\nkHz Variables for {system}:")
    vars_info = data.get_khz_variables(system)
    print(f"  Analog ({len(vars_info['analog'])}):")
    for var in vars_info["analog"][:10]:
        print(f"    {var}")
    if len(vars_info["analog"]) > 10:
        print(f"    ... and {len(vars_info['analog']) - 10} more")
    print(f"  Digital ({len(vars_info['digital'])}):")
    for var in vars_info["digital"][:10]:
        print(f"    {var}")
    if len(vars_info["digital"]) > 10:
        print(f"    ... and {len(vars_info['digital']) - 10} more")


def run_show_rms_vars(data: HybridData, system: str) -> None:
    """Handle --rms-vars command."""
    print(f"\nRMS Variables for {system}:")
    try:
        vars_info = data.get_rms_variables(system)
        print(f"  Analog ({len(vars_info['analog'])}):")
        for var in vars_info["analog"][:10]:
            print(f"    {var}")
        if len(vars_info["analog"]) > 10:
            print(f"    ... and {len(vars_info['analog']) - 10} more")
        print(f"  Digital ({len(vars_info['digital'])}):")
        for var in vars_info["digital"][:10]:
            print(f"    {var}")
        if len(vars_info["digital"]) > 10:
            print(f"    ... and {len(vars_info['digital']) - 10} more")
    except Exception as e:
        log_exception(
            "Error showing kHz variables", e, use_print=True, include_traceback=False
        )
        print(f"  Error at {format_exception_location()}: {e}")


def parse_hours(hours_str: str) -> list:
    """
    Parse comma-separated hours string.

    Parameters
    ----------
    hours_str : str
        Comma-separated hours (e.g., "0,1,2")

    Returns
    -------
    list of int
        List of hour integers

    Raises
    ------
    ValueError
        If the hours string is invalid
    """
    return [int(h.strip()) for h in hours_str.split(",")]


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging level
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging_config = {
        "level": log_level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    if args.log_file:
        logging_config["filename"] = args.log_file
        logging_config["filemode"] = "a"
    logging.basicConfig(**logging_config)
    logger.setLevel(log_level)

    # List dates
    if args.list_dates:
        run_list_dates(args)
        return

    # Require date for other operations
    if not args.date:
        parser.print_help()
        return

    # Create HybridData instance
    try:
        data = HybridData(
            args.base_dir,
            args.date,
            fepc_system=args.fepc_system,
            endian=args.endian,
        )
    except Exception as e:
        log_exception(
            "Error creating HybridData", e, use_print=True, include_traceback=True
        )
        return

    # Show summary
    data.print_summary()

    # Show kHz variables
    if args.khz_vars:
        run_show_khz_vars(data, args.khz_vars)

    # Show RMS variables
    if args.rms_vars:
        run_show_rms_vars(data, args.rms_vars)

    # Parse hours if provided
    hours = None
    if args.hours:
        try:
            hours = parse_hours(args.hours)
        except ValueError:
            print(
                f"Error: Invalid hours format '{args.hours}'. Use comma-separated integers."
            )
            return

    # Plot kHz variable(s)
    if args.plot_khz:
        if not args.fepc_system:
            print("Error: --fepc-system is required for plotting")
            return
        try:
            # Parse comma-separated variables
            variables = [v.strip() for v in args.plot_khz.split(",")]

            if len(variables) == 1:
                # Single variable - use original method
                print(f"\nPlotting kHz variable: {variables[0]}")
                data.plot_khz_variable(
                    args.fepc_system,
                    variables[0],
                    hours=hours,
                    apply_calib=not args.no_calib,
                    save=args.save,
                    remove_outliers_method=args.remove_outliers,
                    outlier_threshold=args.outlier_threshold,
                    outlier_window=args.outlier_window,
                )
            else:
                # Multiple variables - use new multi-variable method
                print(
                    f"\nPlotting kHz variables: {', '.join(variables)} (layout: {args.layout})"
                )
                data.plot_khz_variables(
                    args.fepc_system,
                    variables,
                    hours=hours,
                    apply_calib=not args.no_calib,
                    save=args.save,
                    remove_outliers_method=args.remove_outliers,
                    outlier_threshold=args.outlier_threshold,
                    outlier_window=args.outlier_window,
                    layout=args.layout,
                )
        except ValueError as e:
            print(
                f"Value error plotting kHz variable at {format_exception_location()}: {e}"
            )
            return
        except Exception as e:
            log_exception(
                "Error plotting kHz variable", e, use_print=True, include_traceback=True
            )
            return

    # Plot RMS variable(s)
    if args.plot_rms:
        if not args.fepc_system:
            print("Error: --fepc-system is required for plotting")
            return
        try:
            # Parse comma-separated variables
            variables = [v.strip() for v in args.plot_rms.split(",")]

            if len(variables) == 1:
                # Single variable - use original method
                print(f"\nPlotting RMS variable: {variables[0]}")
                data.plot_rms_variable(
                    args.fepc_system,
                    variables[0],
                    save=args.save,
                )
            else:
                # Multiple variables - use new multi-variable method
                print(
                    f"\nPlotting RMS variables: {', '.join(variables)} (layout: {args.layout})"
                )
                data.plot_rms_variables(
                    args.fepc_system,
                    variables,
                    save=args.save,
                    layout=args.layout,
                )
        except Exception as e:
            log_exception(
                "Error plotting RMS variable", e, use_print=True, include_traceback=True
            )

    # Plot both kHz and RMS
    if args.plot_both:
        if not args.fepc_system:
            print("Error: --fepc-system is required for plotting")
            return
        try:
            rms_var = args.rms_var if args.rms_var else args.plot_both
            print(f"\nPlotting kHz ({args.plot_both}) and RMS ({rms_var})")
            data.plot_khz_with_rms(
                args.fepc_system,
                args.plot_both,
                rms_variable=rms_var,
                hours=hours,
                apply_calib=not args.no_calib,
                save=args.save,
            )
        except ValueError as e:
            print(
                f"Value error plotting kHz variable at {format_exception_location()}: {e}"
            )
            return
        except Exception as e:
            log_exception(
                "Error plotting kHz with RMS", e, use_print=True, include_traceback=True
            )


if __name__ == "__main__":
    main()
