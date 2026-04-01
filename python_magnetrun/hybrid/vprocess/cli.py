"""
VProcess Command-Line Interface
================================

Unified CLI for VProcess data operations.

Usage:
    python cli.py validate <file>
    python cli.py plot <file> --vars VAR1 VAR2
    python cli.py batch --dir <directory> --output merged.csv
    python cli.py info <file>
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)


def cmd_info(args):
    """Display file information."""
    from .vprocess_reader import VProcessFileReader

    reader = VProcessFileReader(args.file)
    reader.parse_header()
    reader.print_summary()


def cmd_validate(args):
    """Validate VProcess file."""
    from .validate import validate_vprocess_file

    results = validate_vprocess_file(args.file, check_data=args.check_data, verbose=not args.quiet)

    return 0 if results["valid"] else 1


def cmd_plot(args):
    """Plot VProcess data."""
    from .plot_vprocess import (
        plot_comparison,
        plot_heatmap,
        plot_overview,
        plot_variables,
    )

    if args.heatmap:
        plot_heatmap(
            args.file,
            variables=args.vars,
            max_vars=args.max_vars,
            save_path=args.save,
            show=not args.no_show,
        )
    elif args.compare:
        if len(args.compare) != 2:
            logger.error("--compare requires exactly 2 variables")
            return 1
        plot_comparison(
            args.file,
            args.compare[0],
            args.compare[1],
            save_path=args.save,
            show=not args.no_show,
        )
    elif args.overview:
        plot_overview(
            args.file,
            max_vars=args.max_vars,
            save_path=args.save,
            show=not args.no_show,
        )
    elif args.vars:
        plot_variables(
            args.file,
            args.vars,
            save_path=args.save,
            show=not args.no_show,
            layout=args.layout,
        )
    else:
        logger.error("Specify --vars, --overview, --compare, or --heatmap")
        return 1

    return 0


def cmd_batch(args):
    """Batch process VProcess files."""
    from .batch import (
        analyze_batch,
        export_data,
        find_vprocess_files,
        get_common_variables,
        process_batch,
    )

    # Find files
    file_list = find_vprocess_files(args.dir, args.pattern, args.recursive)

    if not file_list:
        logger.error(f"No files found in {args.dir} matching {args.pattern}")
        return 1

    logger.info(f"Found {len(file_list)} files")

    # List common variables
    if args.list_common_vars:
        common_vars = get_common_variables(file_list, verbose=not args.quiet)
        logger.info(f"Common variables ({len(common_vars)}):")
        for var in common_vars:
            logger.info(f"  - {var}")
        return 0

    # Analyze files
    if args.analyze:
        summary_df = analyze_batch(file_list, output_file=args.output)
        logger.info("File Analysis Summary:")
        logger.info(f"{summary_df.to_string(index=False)}")
        return 0

    # Process and merge
    if args.merge or args.output:
        df = process_batch(file_list, selected_vars=args.vars, merge=True, verbose=not args.quiet)

        logger.info(f"\nMerged data shape: {df.shape}")
        if len(df) > 0:
            logger.info(f"Time range: {df.index[0]} to {df.index[-1]}")
            duration = (df.index[-1] - df.index[0]).total_seconds()
            logger.info(f"Duration: {duration:.1f} seconds ({duration / 3600:.2f} hours)")

        if args.output:
            export_data(df, args.output, args.format)

        return 0

    logger.error("Specify --merge, --output, --analyze, or --list-common-vars")
    return 1


def cmd_test(args):
    """Run tests."""
    from .test import create_mock_vprocess_file, run_all_tests

    if args.create_mock:
        create_mock_vprocess_file(
            filepath=args.output,
            n_samples=args.samples,
            n_analog=args.analog,
            n_digital=args.digital,
        )
        return 0

    # Create or use test file
    if args.file:
        test_file = args.file
    else:
        test_file = create_mock_vprocess_file(
            filepath="test_data.vprocess", n_samples=100, n_analog=5, n_digital=1
        )

    success = run_all_tests(test_file)

    # Cleanup if we created the file
    if not args.file:
        Path(test_file).unlink(missing_ok=True)

    return 0 if success else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VProcess Data Tools - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info command
    info_parser = subparsers.add_parser("info", help="Display file information")
    info_parser.add_argument("file", help="VProcess file")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate file")
    validate_parser.add_argument("file", help="VProcess file")
    validate_parser.add_argument("--check-data", action="store_true", help="Check data integrity")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot data")
    plot_parser.add_argument("file", help="VProcess file")
    plot_parser.add_argument("--vars", nargs="+", help="Variables to plot")
    plot_parser.add_argument("--overview", action="store_true", help="Plot overview")
    plot_parser.add_argument(
        "--compare", nargs=2, metavar=("VAR1", "VAR2"), help="Compare variables"
    )
    plot_parser.add_argument("--heatmap", action="store_true", help="Correlation heatmap")
    plot_parser.add_argument("--max-vars", type=int, default=10, help="Max variables (default: 10)")
    plot_parser.add_argument(
        "--layout",
        choices=["subplots", "overlay"],
        default="subplots",
        help="Plot layout",
    )
    plot_parser.add_argument("--save", "-s", help="Save to file")
    plot_parser.add_argument("--no-show", action="store_true", help="Don't display plot")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process files")
    batch_parser.add_argument("--dir", required=True, help="Directory with files")
    batch_parser.add_argument("--pattern", default="*.vprocess", help="File pattern")
    batch_parser.add_argument("-r", "--recursive", action="store_true", help="Search recursively")
    batch_parser.add_argument("--vars", nargs="+", help="Variables to extract")
    batch_parser.add_argument("--merge", action="store_true", help="Merge files")
    batch_parser.add_argument("--output", "-o", help="Output file")
    batch_parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "hdf5", "parquet", "excel"],
        help="Output format",
    )
    batch_parser.add_argument(
        "--list-common-vars", action="store_true", help="List common variables"
    )
    batch_parser.add_argument("--analyze", action="store_true", help="Analyze files")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--file", help="Test specific file")
    test_parser.add_argument("--create-mock", action="store_true", help="Create mock file")
    test_parser.add_argument("--output", default="mock_data.vprocess", help="Mock file output")
    test_parser.add_argument("--samples", type=int, default=3600, help="Number of samples")
    test_parser.add_argument("--analog", type=int, default=10, help="Analog variables")
    test_parser.add_argument("--digital", type=int, default=2, help="Digital variables")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")

    # Execute command
    if not args.command:
        parser.print_help()
        return 1

    command_map = {
        "info": cmd_info,
        "validate": cmd_validate,
        "plot": cmd_plot,
        "batch": cmd_batch,
        "test": cmd_test,
    }

    try:
        return command_map[args.command](args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except (OSError, ValueError, RuntimeError) as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
