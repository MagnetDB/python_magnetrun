"""
Command-line interface for magnetrun analysis.

This module provides:
- Command-line argument parsing
- Main entry point for the analysis workflow

Logging infrastructure (setup_logging, ColoredFormatter, JSONFormatter, LogConfig,
get_logger, set_log_level, ProgressTracker, timed_operation, LogContext) is shared
across all entry points via python_magnetrun.log_utils.

Usage::

    python -m python_magnetrun.analysis.cli input1.tdms input2.tdms --show --save

    # With verbose logging:
    python -m python_magnetrun.analysis.cli input.tdms --debug --log-file analysis.log

    # With JSON logging:
    python -m python_magnetrun.analysis.cli input.tdms --json-log analysis.json

Example programmatic usage::

    from python_magnetrun.log_utils import (
        setup_logging,
        log_exception,
        format_exception_location,
        LogContext,
        timed_operation,
        ProgressTracker,
    )

    # Setup logging
    logger = setup_logging(debug=True, log_file="analysis.log")

    # Exception handling with detailed logging
    try:
        risky_operation()
    except Exception as e:
        log_exception(logger, "Operation failed", e)
        # Or get just the location
        location = format_exception_location()
        logger.error(f"Error at {location}: {e}")

    # Use timing context
    with timed_operation("Loading data"):
        df = load_data(files)

    # Track progress
    tracker = ProgressTracker(total=100, description="Processing files")
    for i in range(100):
        # ... do work ...
        tracker.update()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from natsort import natsorted

from ..log_utils import (
    ProgressTracker,
    get_logger,
    setup_logging,
    timed_operation,
)
from ..utils.files import expand_input_files
from .args import args_to_processing_config, parse_arguments
from .processing import print_record_summary, process_overview_file

# =============================================================================
# Main entry point
# =============================================================================


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the analysis CLI.

    Parameters
    ----------
    args : list of str, optional
        Command-line arguments (for testing)

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors)
    """
    parsed_args = parse_arguments(args)

    # Setup logging
    setup_logging(
        debug=parsed_args.debug,
        log_file=parsed_args.log_file,
        json_file=parsed_args.json_log,
        use_colors=not parsed_args.no_color,
        quiet=parsed_args.quiet,
    )
    logger = get_logger("analysis.cli")

    logger.info("Starting magnetrun analysis")
    logger.info(f"Arguments: {parsed_args}")
    logger.debug(f"input_file: {parsed_args.input_file}")  # noqa: F823

    housing = parsed_args.housing
    try:
        from .config import get_housing_config
        from .metrics import (
            calc_correlation,
            calc_euclidean,
            calc_mape,
            compute_dtw_distance,
        )
        from .plotting import (
            estimate_downsample_percent,
            plot_data,
        )

        # Convert args to processing config
        config = args_to_processing_config(parsed_args)

        # Expand glob patterns and search data directories for bare filenames.
        datadir = {".tdms": str(config.pigbrother_datadir)}
        input_files = cast(
            list[str],
            natsorted(
                expand_input_files(parsed_args.input_file, datadir, parsed_args.housing)
            ),
        )
        logger.debug(f"input_files: {input_files}")
        logger.info(f"Processing {len(input_files)} input files")

        # Create output directory
        if parsed_args.save:
            parsed_args.output_dir.mkdir(parents=True, exist_ok=True)

        # Process each file
        tracker = ProgressTracker(
            total=len(input_files),
            description="Processing files",
            log_interval=1,
        )

        results = []
        for input_file in input_files:
            with timed_operation(f"Processing {Path(input_file).name}", logger):
                try:
                    record = process_overview_file(
                        input_file, config, dry_run=parsed_args.dry_run
                    )
                    print_record_summary(record)

                    results.append(record)
                    logger.info(
                        f"Processed {record.filename}: housing={record.housing}, duration={record.duration:.1f}s,"
                        f" has_pupitre={record.has_data('pupitre')}"
                        f" has_incidents={record.has_data('incidents')}"
                        f" has_hybrid={record.has_data('hybrid_kHz')}"
                        f" has_hybrid_incidents={record.has_data('hybrid_trigger')}"
                    )

                    # Skip further processing if dry run
                    if parsed_args.dry_run:
                        continue

                    # Get housing config for channel mappings
                    housing = (
                        Path(input_file).name.split("_")[0]
                        if housing == "notdefined"
                        else housing
                    )
                    print(f"Determined housing: {housing} from filename: {input_file}")

                    # instead get_housing_config from input_file
                    from .config import AnalysisConfig

                    housing_config = get_housing_config(housing)
                    analysis_cfg = AnalysisConfig.for_housing(housing)
                    channel_map = analysis_cfg.channels

                    # Build setpoint→actual and setpoint→pupitre dicts from ChannelMapping.
                    channels_dict = channel_map.to_dict()
                    pupitre_dict = {
                        record.housing: {
                            channel_map.get_setpoint_channel(g): (
                                housing_config.reference_gr1_current
                                if g == "GR1"
                                else housing_config.reference_gr2_current
                            )
                            for g in channel_map.groups()
                        }
                    }

                    # Get DataFrames
                    df_overview = record.get_overview()
                    print(f"df_overview columns: {df_overview.columns.tolist()}")
                    df_archive = record.get_archive()
                    print(f"df_archive columns: {df_archive.columns.tolist()}")
                    df_pupitre = record.get_pupitre()
                    print(f"df_pupitre columns: {df_pupitre.columns.tolist()}")
                    df_incidents = record.get_incidents()
                    for key, _dfs in df_incidents.items():
                        print(f"df_incidents[{key}]:")
                        for _df in _dfs:
                            print(f"columns: {_df.columns.tolist()}")
                    logger.info("get database done")
                    logger.info(f"df_archive: {df_archive.head()}")

                    # Determine keys to analyze: all setpoint channels
                    keys = [
                        channel_map.get_setpoint_channel(g)
                        for g in channel_map.groups()
                    ]

                    # Process each key
                    for key in keys:
                        if key not in df_overview.columns:
                            logger.warning(f"Key {key} not found in overview data")
                            continue

                        pupitre_key = pupitre_dict[record.housing].get(key)

                        # === PLOTTING ===
                        if parsed_args.show or parsed_args.save:
                            with timed_operation(f"Plotting {key}", logger):
                                # Estimate downsampling if not specified
                                downsample_pct = parsed_args.downsample
                                if downsample_pct == 100.0 and len(df_overview) > 10000:
                                    downsample_pct = estimate_downsample_percent(
                                        len(df_overview), target_points=10000
                                    )
                                    logger.info(
                                        f"Auto-downsampling to {downsample_pct:.1f}% for plotting"
                                    )

                                # Determine output path
                                output_path = None
                                if parsed_args.save:
                                    output_path = (
                                        parsed_args.output_dir
                                        / f"{record.filename}_{key.replace('Courant_', '')}.png"
                                    )

                                # Sync message
                                msg = "(nosync)"
                                if (
                                    config.synchronize
                                    and "timeshift_seconds" in record.sync_info
                                ):
                                    shift = record.sync_info["timeshift_seconds"]
                                    msg = f"(sync: {shift:.2f}s)"

                                # Create plot
                                plot_data(
                                    df_overview=df_overview,
                                    df_archive=df_archive,
                                    df_pupitre=df_pupitre,
                                    df_incidents=df_incidents,
                                    channels_dict=channels_dict,
                                    pupitre_dict=pupitre_dict,
                                    housing=record.housing,
                                    tkey=parsed_args.tkey,
                                    key=key,
                                    title=record.filename,
                                    msg=msg,
                                    show=parsed_args.show,
                                    save=parsed_args.save,
                                    output_path=(
                                        str(output_path) if output_path else None
                                    ),
                                    downsample_percent=downsample_pct,
                                )

                                if output_path:
                                    logger.info(f"Saved plot to {output_path}")

                        # === DISTANCE METRICS ===
                        if (
                            parsed_args.distance
                            and pupitre_key
                            and record.has_data("pupitre")
                        ):
                            if pupitre_key in df_pupitre.columns:
                                with timed_operation(
                                    f"Computing metrics for {key}", logger
                                ):
                                    # Get aligned time series
                                    # Use overview as reference

                                    # Resample pupitre to match overview length
                                    import numpy as np

                                    series1 = np.asarray(df_overview[key], dtype=float)

                                    pupitre_values = np.asarray(
                                        df_pupitre[pupitre_key], dtype=float
                                    )
                                    if len(pupitre_values) != len(series1):
                                        # Simple resampling by interpolation
                                        x_orig = np.linspace(0, 1, len(pupitre_values))
                                        x_new = np.linspace(0, 1, len(series1))
                                        series2 = np.interp(
                                            x_new, x_orig, pupitre_values
                                        )
                                    else:
                                        series2 = pupitre_values

                                    # Compute metrics
                                    euclidean = calc_euclidean(series1, series2)
                                    mape = calc_mape(series1, series2)
                                    correlation = calc_correlation(series1, series2)

                                    logger.info(
                                        f"Metrics for {key} vs {pupitre_key}: "
                                        f"Euclidean={euclidean:.4f}, MAPE={mape:.2f}%, Correlation={correlation:.4f}"
                                    )

                                    # Store in record
                                    record.metrics[key] = {
                                        "euclidean": euclidean,
                                        "mape": mape,
                                        "correlation": correlation,
                                    }

                                    # DTW (can be slow for large datasets)
                                    if len(series1) <= 5000:
                                        dtw_result = compute_dtw_distance(
                                            series1, series2
                                        )
                                        # print(dtw_result.distance)           # The DTW distance
                                        # print(dtw_result.path)               # The warping path
                                        # print(dtw_result.normalized_distance) # Normalized by length
                                        # print(dtw_result.similarity_score)    # Distance per path step
                                        logger.info(
                                            f"DTW distance for {key}: {dtw_result.similarity_score:.4f}"
                                        )
                                        record.metrics[key][
                                            "dtw"
                                        ] = dtw_result.similarity_score
                                    else:
                                        logger.info(
                                            f"Skipping DTW for {key} (dataset too large: {len(series1)} points)"
                                        )
                            else:
                                logger.warning(
                                    f"Pupitre key {pupitre_key} not found for distance metrics"
                                )

                except (OSError, ValueError, KeyError, RuntimeError) as e:
                    logger.error(f"Failed to process {input_file}: {e}")
                    if parsed_args.debug:
                        logger.exception("Full traceback:")

            tracker.update()

        tracker.finish()

        # === FINAL SUMMARY ===
        successful = len(results)
        failed = len(input_files) - successful

        logger.info(f"Analysis complete: {successful} successful, {failed} failed")

        # Print metrics summary if computed
        if parsed_args.distance and results:
            logger.info("=== Metrics Summary ===")
            for record in results:
                if record.metrics:
                    logger.info(f"File: {record.filename}")
                    for key, metrics in record.metrics.items():
                        logger.info(
                            f"  {key}: Euclidean={metrics.get('euclidean', 0):.4f}, "
                            f"MAPE={metrics.get('mape', 0):.2f}%, "
                            f"Corr={metrics.get('correlation', 0):.4f}"
                            + (
                                f", DTW={metrics['dtw']:.4f}"
                                if "dtw" in metrics
                                else ""
                            )
                        )

        return 0 if failed == 0 else 1

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except (ImportError, OSError, RuntimeError) as e:
        logger.exception(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
