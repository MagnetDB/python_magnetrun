"""
VProcess Command-Line Interface
================================

Unified CLI for VProcess data operations.

Usage:
    python vprocess_cli.py validate <file>
    python vprocess_cli.py plot <file> --vars VAR1 VAR2
    python vprocess_cli.py batch --dir <directory> --output merged.csv
    python vprocess_cli.py info <file>
"""

import logging
import sys
from pathlib import Path

from python_magnetrun.log_utils import BARE_FORMAT, setup_logging

_examples_dir = str(Path(__file__).parent)
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)

from vprocess_args import create_argument_parser  # noqa: E402

# Setup logger
logger = logging.getLogger(__name__)


def cmd_info(args):
    """Display file information."""
    from python_magnetrun.hybrid.vprocess import VProcessFileReader

    reader = VProcessFileReader(args.file)
    reader.parse_header()
    reader.print_summary()


def cmd_validate(args):
    """Validate VProcess file."""
    from python_magnetrun.hybrid.vprocess.validate import validate_vprocess_file

    results = validate_vprocess_file(
        args.file, check_data=args.check_data, verbose=not args.quiet
    )

    return 0 if results["valid"] else 1


def cmd_plot(args):
    """Plot VProcess data."""
    from python_magnetrun.hybrid.vprocess.plot_vprocess import (
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
    from vprocess_batch import (
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
        df = process_batch(
            file_list, selected_vars=args.vars, merge=True, verbose=not args.quiet
        )

        logger.info(f"\nMerged data shape: {df.shape}")
        if len(df) > 0:
            logger.info(f"Time range: {df.index[0]} to {df.index[-1]}")
            duration = (df.index[-1] - df.index[0]).total_seconds()
            logger.info(
                f"Duration: {duration:.1f} seconds ({duration / 3600:.2f} hours)"
            )

        if args.output:
            export_data(df, args.output, args.format)

        return 0

    logger.error("Specify --merge, --output, --analyze, or --list-common-vars")
    return 1


def cmd_test(args):
    """Run tests."""
    from python_magnetrun.hybrid.vprocess.test import create_mock_vprocess_file, run_all_tests

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
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = (
        logging.WARNING
        if args.quiet
        else (logging.DEBUG if args.verbose else logging.INFO)
    )
    setup_logging(level=log_level, fmt=BARE_FORMAT)

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
