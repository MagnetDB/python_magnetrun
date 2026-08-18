from python_magnetrun.analysis.processing import (
        process_overview_file,
        ProcessingConfig,
    )
import logging
from python_magnetrun.log_utils import get_logger, setup_logging
from pathlib import Path

logger = get_logger(__name__)

def main() -> None:
    import argparse
    from python_magnetrun.cli_args import validate_file_extension

    from tabulate import tabulate

    from python_magnetrun.analysis.field_comparison import (
        AliasedField,
        compare_field,
        compute_reference_lag,
        discover_pupitre_pigbrother_fields,
        print_comparison_summary,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=validate_file_extension([".tdms"]),
        help="Overview file to process",
        default="M9_Overview_241106-0935.tdms"
    )
    parser.add_argument(
        "--housing",
        default=None,
        type=str,
        help="Set housing name",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to save plots (PNG) to; created if missing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="path to log file (if not specified, logs to console)",
    )
    parser.add_argument("--show", help="enable show mode", action="store_true")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    setup_logging(level=log_level, log_file=args.log_file)
    logger.debug(f"Parsed arguments: {args}")

    plot_enabled = args.show or args.output_dir is not None
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.housing is None:
        housing = args.input_file.split('_')[0]
        print(f'Housing: {housing}')
        
    from python_magnetrun.analysis.field_comparison import (
        compare_all_fields,
        print_comparison_summary,
    )

    record = process_overview_file(args.input_file, ProcessingConfig())
    print(f'loading record: overview={record.sources.overview}, archive={record.sources.archive},  pupitre={record.sources.pupitre}')

    for method in ["none", "resample_1s", "interpolated"]:
        method_output_dir = None
        if args.output_dir:
            method_output_dir = str(Path(args.output_dir) / method)
            Path(method_output_dir).mkdir(parents=True, exist_ok=True)

        results = compare_all_fields(
            record,
            lag_method=method,
            plot=plot_enabled,
            output_dir=method_output_dir,
            show=args.show,
        )
        # print(f'results: {results}')
        print_comparison_summary(results)
        if plot_enabled and method_output_dir:
            print(f"Saved comparison plots to {method_output_dir}/")

if __name__ == "__main__":
    main()
