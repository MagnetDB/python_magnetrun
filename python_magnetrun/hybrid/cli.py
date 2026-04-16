"""
Command-line interface for hybrid magnet data (kHz, RMS, Trigger)

This module provides CLI tools for reading and plotting hybrid magnet data
from FEPC acquisition systems.

Usage:
    python -m python_magnetrun.hybrid.cli --help
    python -m python_magnetrun.hybrid.cli --base-dir /data/hybrid --date 2025-01-06
"""

import logging

from ..log_utils import setup_logging
from .args import create_parser
from .hybrid_data import HybridData
from .utils import format_exception_location, list_available_dates, log_exception

# Setup logger
logger = logging.getLogger(__name__)


def run_list_dates(args) -> None:
    """Handle --list-dates command."""
    logger.info("Available dates:")
    for data_type in ["kHz", "rms", "trigger"]:
        dates = list_available_dates(args.base_dir, data_type)
        if dates:
            logger.info(
                f"  {data_type}: {', '.join(dates[:5])}"
                + (f" ... ({len(dates)} total)" if len(dates) > 5 else "")
            )


def run_show_khz_vars(data: HybridData, system: str) -> None:
    """Handle --khz-vars command."""
    logger.info(f"\nkHz Variables for {system}:")
    vars_info = data.get_khz_variables(system)
    logger.info(f"  Analog ({len(vars_info['analog'])}):")
    for var in vars_info["analog"][:10]:
        logger.info(f"    {var}")
    if len(vars_info["analog"]) > 10:
        logger.info(f"    ... and {len(vars_info['analog']) - 10} more")
    logger.info(f"  Digital ({len(vars_info['digital'])}):")
    for var in vars_info["digital"][:10]:
        logger.info(f"    {var}")
    if len(vars_info["digital"]) > 10:
        logger.info(f"    ... and {len(vars_info['digital']) - 10} more")


def run_show_rms_vars(data: HybridData, system: str) -> None:
    """Handle --rms-vars command."""
    logger.info(f"\nRMS Variables for {system}:")
    try:
        vars_info = data.get_rms_variables(system)
        logger.info(f"  Analog ({len(vars_info['analog'])}):")
        for var in vars_info["analog"][:10]:
            logger.info(f"    {var}")
        if len(vars_info["analog"]) > 10:
            logger.info(f"    ... and {len(vars_info['analog']) - 10} more")
        logger.info(f"  Digital ({len(vars_info['digital'])}):")
        for var in vars_info["digital"][:10]:
            logger.info(f"    {var}")
        if len(vars_info["digital"]) > 10:
            logger.info(f"    ... and {len(vars_info['digital']) - 10} more")
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        log_exception(
            "Error showing kHz variables", e, use_print=True, include_traceback=False
        )
        logger.error(f"  Error at {format_exception_location()}: {e}")


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
    setup_logging(level=log_level, log_file=args.log_file if args.log_file else None)
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
    except (OSError, ValueError, RuntimeError) as e:
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
            logger.error(
                f"Error: Invalid hours format '{args.hours}'. Use comma-separated integers."
            )
            return

    # Plot kHz variable(s)
    if args.plot_khz:
        if not args.fepc_system:
            logger.error("Error: --fepc-system is required for plotting")
            return
        try:
            # Parse comma-separated variables
            variables = [v.strip() for v in args.plot_khz.split(",")]

            if len(variables) == 1:
                # Single variable - use original method
                logger.info(f"\nPlotting kHz variable: {variables[0]}")
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
                logger.info(
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
            logger.error(
                f"Value error plotting kHz variable at {format_exception_location()}: {e}"
            )
            return
        except (OSError, RuntimeError) as e:
            log_exception(
                "Error plotting kHz variable", e, use_print=True, include_traceback=True
            )
            return

    # Plot RMS variable(s)
    if args.plot_rms:
        if not args.fepc_system:
            logger.error("Error: --fepc-system is required for plotting")
            return
        try:
            # Parse comma-separated variables
            variables = [v.strip() for v in args.plot_rms.split(",")]

            if len(variables) == 1:
                # Single variable - use original method
                logger.info(f"\nPlotting RMS variable: {variables[0]}")
                data.plot_rms_variable(
                    args.fepc_system,
                    variables[0],
                    save=args.save,
                )
            else:
                # Multiple variables - use new multi-variable method
                logger.info(
                    f"\nPlotting RMS variables: {', '.join(variables)} (layout: {args.layout})"
                )
                data.plot_rms_variables(
                    args.fepc_system,
                    variables,
                    save=args.save,
                    layout=args.layout,
                )
        except (OSError, ValueError, RuntimeError) as e:
            log_exception(
                "Error plotting RMS variable", e, use_print=True, include_traceback=True
            )

    # Plot both kHz and RMS
    if args.plot_both:
        if not args.fepc_system:
            logger.error("Error: --fepc-system is required for plotting")
            return
        try:
            rms_var = args.rms_var if args.rms_var else args.plot_both
            logger.info(f"\nPlotting kHz ({args.plot_both}) and RMS ({rms_var})")
            data.plot_khz_with_rms(
                args.fepc_system,
                args.plot_both,
                rms_variable=rms_var,
                hours=hours,
                apply_calib=not args.no_calib,
                save=args.save,
            )
        except ValueError as e:
            logger.error(
                f"Value error plotting kHz variable at {format_exception_location()}: {e}"
            )
            return
        except (OSError, RuntimeError) as e:
            log_exception(
                "Error plotting kHz with RMS", e, use_print=True, include_traceback=True
            )


if __name__ == "__main__":
    main()
