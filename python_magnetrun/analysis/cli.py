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
from .args import (
    args_to_downsample_config,
    args_to_processing_config,
    parse_arguments,
)
from .processing import print_record_summary, process_overview_file

# =============================================================================
# Helpers
# =============================================================================


def _setup_logging(parsed_args):
    """Configure logging from parsed args and return the module logger."""
    if parsed_args.debug:
        _log_level = "DEBUG"
        _quiet = False
    elif parsed_args.quiet:
        _log_level = "WARNING"
        _quiet = True
    else:
        _log_level = parsed_args.log_level
        _quiet = False
    setup_logging(
        level=_log_level,
        debug=False,
        log_file=parsed_args.log_file,
        json_file=parsed_args.json_log,
        use_colors=not parsed_args.no_color,
        quiet=_quiet,
    )
    logger = get_logger("analysis.cli")
    logger.info("Starting magnetrun analysis")
    logger.info(f"Arguments: {parsed_args}")
    logger.debug(f"input_file: {parsed_args.input_file}")  # noqa: F823
    return logger


def _collect_input_files(parsed_args, config) -> list[str]:
    """Expand glob patterns / bare filenames and return a sorted file list."""
    datadir = {".tdms": str(config.pigbrother_datadir)}
    return cast(
        list[str],
        natsorted(
            expand_input_files(parsed_args.input_file, datadir, parsed_args.housing)
        ),
    )


def _load_records(input_files, config, parsed_args, housing, logger):
    """Process each input file and accumulate DataFrames + channel config.

    Returns
    -------
    (results, all_dfs, channels_dict, pupitre_dict, hybrid_dict, keys, housing)
    """
    from .config import AnalysisConfig, get_housing_config

    results = []
    all_dfs: list = []
    summary_rows: list[dict] = []
    channels_dict: dict = {}
    pupitre_dict: dict = {}
    hybrid_dict: dict = {}
    keys: list[str] = []

    tracker = ProgressTracker(
        total=len(input_files),
        description="Processing files",
        log_interval=1,
    )

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

                if parsed_args.dry_run:
                    tracker.update()
                    continue

                housing = (
                    Path(input_file).name.split("_")[0]
                    if housing == "notdefined"
                    else housing
                )
                logger.info(
                    f"Determined housing: {housing} from filename: {input_file}"
                )

                housing_config = get_housing_config(housing)
                analysis_cfg = AnalysisConfig.for_housing(housing)
                channel_map = analysis_cfg.channels

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

                hybrid_dict = {
                    record.housing: {
                        channel_map.get_setpoint_channel(g): (
                            housing_config.reference_gr1_hybrid
                            if g == "GR1"
                            else housing_config.reference_gr2_hybrid
                        )
                        for g in channel_map.groups()
                    }
                }

                df_overview = record.get_overview()
                logger.info(f"df_overview columns: {df_overview.columns.tolist()}")
                df_archive = record.get_archive()
                logger.info(f"df_archive columns: {df_archive.columns.tolist()}")
                df_pupitre = record.get_pupitre()
                logger.info(f"df_pupitre columns: {df_pupitre.columns.tolist()}")
                df_incidents = record.get_incidents()
                for key, _dfs in df_incidents.items():
                    logger.info(f"df_incidents[{key}]:")
                    for _df in _dfs:
                        logger.info(f"columns: {_df.columns.tolist()}")
                logger.info("get database done")
                logger.info(f"df_archive: {df_archive.head()}")

                df_hybrid = (
                    record.get_hybrid_kHz() if record.has_data("hybrid_kHz") else None
                )
                logger.info(
                    f"df_hybrid: {df_hybrid.head() if df_hybrid is not None else 'None'}"
                )
                df_hybrid_incidents = (
                    (
                        record.get_hybrid_incidents()
                        if record.has_data("hybrid_trigger")
                        else None
                    )
                    if housing == "M8"
                    else None
                )
                logger.info(
                    f"df_hybrid_incidents: {df_hybrid_incidents if df_hybrid_incidents is not None else 'None'}"
                )

                all_dfs.append(
                    (
                        df_overview,
                        df_archive,
                        df_pupitre,
                        df_incidents,
                        df_hybrid,
                        df_hybrid_incidents,
                    )
                )
                if record.sources:
                    s = record.sources
                    summary_rows.append(
                        {
                            "filename": record.filename,
                            "overview": ", ".join(s.overview),
                            "archive": ", ".join(s.archive),
                            "pupitre": ", ".join(s.pupitre),
                            "default": ", ".join(s.default),
                            "trigger": ", ".join(s.trigger),
                            "spike": ", ".join(s.spike),
                            "hybrid_kHz": ", ".join(s.hybrid_kHz),
                            "hybrid_rms": ", ".join(s.hybrid_rms),
                            "hybrid_trigger": ", ".join(s.hybrid_trigger),
                            "hybrid_vprocess": ", ".join(s.hybrid_vprocess),
                            "pigbrother_runlog": ", ".join(s.pigbrother_runlog),
                            "pupitre_runlog": ", ".join(s.pupitre_runlog),
                        }
                    )
                keys = [
                    channel_map.get_setpoint_channel(g) for g in channel_map.groups()
                ]

            except (OSError, ValueError, KeyError, RuntimeError) as e:
                logger.error(f"Failed to process {input_file}: {e}")
                if parsed_args.debug:
                    logger.exception("Full traceback:")

        tracker.update()

    if summary_rows:
        import os

        import pandas as pd
        from tabulate import tabulate

        raw_df = pd.DataFrame(summary_rows)
        
        def _basenames(v: str) -> str:
            files = [x.strip() for x in v.split(",") if x.strip()]
            return "\n".join(os.path.basename(f) for f in files)

        list_cols = [
            "overview", "archive", "pupitre",
            "default", "trigger", "spike",
            "hybrid_kHz", "hybrid_rms", "hybrid_trigger", "hybrid_vprocess",
            "pigbrother_runlog", "pupitre_runlog",
        ]
        compact: dict = {"filename": raw_df["filename"].tolist()}
        for col in list_cols:
            if col in raw_df.columns:
                compact[col] = raw_df[col].apply(_basenames)
        summary_df = pd.DataFrame(compact)

        print()
        print(tabulate(
            summary_df.values.tolist(),
            headers=summary_df.columns.tolist(),
            tablefmt="rounded_grid",
        ))

        json_path = Path(parsed_args.output_dir) / "summary.json"
        raw_df.to_json(json_path, orient="records", indent=2)
        logger.info(f"Summary saved to {json_path}")

    tracker.finish()
    return results, all_dfs, channels_dict, pupitre_dict, hybrid_dict, keys, housing


def _combine_dataframes(all_dfs: list[tuple]):
    """Concatenate per-file DataFrames into single combined DataFrames.

    Returns
    -------
    (df_overview, df_archive, df_pupitre, df_incidents, df_hybrid, df_hybrid_incidents)
    """
    import pandas as pd

    df_overview = pd.concat([d[0] for d in all_dfs], ignore_index=True)
    df_archive = pd.concat([d[1] for d in all_dfs], ignore_index=True)
    pupitre_list = [d[2] for d in all_dfs if d[2] is not None and not d[2].empty]
    df_pupitre = (
        pd.concat(pupitre_list, ignore_index=True) if pupitre_list else pd.DataFrame()
    )
    df_incidents: dict = {}
    for d in all_dfs:
        for k, v in d[3].items():
            df_incidents.setdefault(k, []).extend(v)
    hybrid_list = [d[4] for d in all_dfs if d[4] is not None and not d[4].empty]
    df_hybrid = (
        pd.concat(hybrid_list, ignore_index=True) if hybrid_list else pd.DataFrame()
    )
    hybrid_inc_list = [d[5] for d in all_dfs if d[5] is not None and not d[5].empty]
    df_hybrid_incidents = (
        pd.concat(hybrid_inc_list, ignore_index=True)
        if hybrid_inc_list
        else pd.DataFrame()
    )
    return (
        df_overview,
        df_archive,
        df_pupitre,
        df_incidents,
        df_hybrid,
        df_hybrid_incidents,
    )


def _run_combined_analysis(
    results,
    all_dfs,
    keys,
    housing,
    channels_dict,
    pupitre_dict,
    hybrid_dict,
    parsed_args,
    config,
    logger,
) -> dict:
    """Concat all DataFrames, then run plotting and metrics once per key.

    Returns
    -------
    combined_metrics : dict
    """
    import numpy as np

    from .metrics import (
        calc_correlation,
        calc_euclidean,
        calc_mape,
        compute_dtw_distance,
    )
    from .plotting import plot_data

    (
        df_overview,
        df_archive,
        df_pupitre,
        df_incidents,
        df_hybrid,
        df_hybrid_incidents,
    ) = _combine_dataframes(all_dfs)
    combined_title = (
        results[0].filename if len(results) == 1 else f"{len(results)} files"
    )
    combined_metrics: dict = {}
    downsample_config = args_to_downsample_config(parsed_args)

    for key in keys:
        if key not in df_overview.columns:
            logger.warning(f"Key {key} not found in overview data")
            continue
        logger.info(f"processing key: {key}")
        pupitre_key = pupitre_dict.get(housing, {}).get(key)
        hybrid_key = hybrid_dict.get(housing, {}).get(key)

        # === PLOTTING ===
        if parsed_args.show or parsed_args.save:
            with timed_operation(
                f"Plotting {key}, synchronize={config.synchronize}, downsample_config={downsample_config!r}",
                logger,
            ):
                output_path = None
                if parsed_args.save:
                    backend_name = getattr(parsed_args, "backend", "matplotlib")
                    suffix = ".html" if "plotly" in backend_name else ".png"
                    output_path = (
                        parsed_args.output_dir
                        / f"{combined_title}_{key.replace('Courant_', '')}{suffix}"
                    )

                msg = "(nosync)"
                if (
                    config.synchronize
                    and results
                    and "timeshift_seconds" in results[0].sync_info
                ):
                    shift = results[0].sync_info["timeshift_seconds"]
                    msg = f"(sync: {shift:.2f}s)"

                plot_data(
                    df_overview=df_overview,
                    df_archive=df_archive,
                    df_pupitre=df_pupitre,
                    df_incidents=df_incidents,
                    channels_dict=channels_dict,
                    pupitre_dict=pupitre_dict,
                    housing=housing,
                    tkey=parsed_args.tkey,
                    key=key,
                    title=combined_title,
                    msg=msg,
                    show=parsed_args.show,
                    save=parsed_args.save,
                    output_path=(str(output_path) if output_path else None),
                    downsample_config=downsample_config,
                    df_hybrid=df_hybrid,
                    df_hybrid_incidents=df_hybrid_incidents,
                    hybrid_dict=hybrid_dict,
                    backend=getattr(parsed_args, "backend", "matplotlib"),
                )

                if output_path:
                    logger.info(f"Saved plot to {output_path}")

        # === DISTANCE METRICS ===
        if (
            parsed_args.distance
            and pupitre_key
            and any(r.has_data("pupitre") for r in results)
        ):
            if pupitre_key in df_pupitre.columns:
                with timed_operation(f"Computing metrics for {key}", logger):
                    series1 = np.asarray(df_overview[key], dtype=float)
                    pupitre_values = np.asarray(df_pupitre[pupitre_key], dtype=float)
                    if len(pupitre_values) != len(series1):
                        x_orig = np.linspace(0, 1, len(pupitre_values))
                        x_new = np.linspace(0, 1, len(series1))
                        series2 = np.interp(x_new, x_orig, pupitre_values)
                    else:
                        series2 = pupitre_values

                    euclidean = calc_euclidean(series1, series2)
                    mape = calc_mape(series1, series2)
                    correlation = calc_correlation(series1, series2)

                    logger.info(
                        f"Metrics for {key} vs {pupitre_key}: "
                        f"Euclidean={euclidean:.4f}, MAPE={mape:.2f}%, Correlation={correlation:.4f}"
                    )

                    combined_metrics[key] = {
                        "euclidean": euclidean,
                        "mape": mape,
                        "correlation": correlation,
                    }

                    if len(series1) <= 5000:
                        dtw_result = compute_dtw_distance(series1, series2)
                        logger.info(
                            f"DTW distance for {key}: {dtw_result.similarity_score:.4f}"
                        )
                        combined_metrics[key]["dtw"] = dtw_result.similarity_score
                    else:
                        logger.info(
                            f"Skipping DTW for {key} (dataset too large: {len(series1)} points)"
                        )
            else:
                logger.warning(
                    f"Pupitre key {pupitre_key} not found for distance metrics"
                )

        # === HYBRID DISTANCE METRICS ===
        if (
            parsed_args.distance
            and hybrid_key
            and any(r.has_data("hybrid_kHz") for r in results)
        ):
            if hybrid_key in df_hybrid.columns:
                with timed_operation(f"Computing hybrid metrics for {key}", logger):
                    series1 = np.asarray(df_overview[key], dtype=float)
                    hybrid_values = np.asarray(df_hybrid[hybrid_key], dtype=float)
                    if len(hybrid_values) != len(series1):
                        x_orig = np.linspace(0, 1, len(hybrid_values))
                        x_new = np.linspace(0, 1, len(series1))
                        series2 = np.interp(x_new, x_orig, hybrid_values)
                    else:
                        series2 = hybrid_values

                    euclidean = calc_euclidean(series1, series2)
                    mape = calc_mape(series1, series2)
                    correlation = calc_correlation(series1, series2)

                    logger.info(
                        f"Hybrid metrics for {key} vs {hybrid_key}: "
                        f"Euclidean={euclidean:.4f}, MAPE={mape:.2f}%, Correlation={correlation:.4f}"
                    )

                    combined_metrics.setdefault(key, {}).update(
                        {
                            "hybrid_euclidean": euclidean,
                            "hybrid_mape": mape,
                            "hybrid_correlation": correlation,
                        }
                    )

                    if len(series1) <= 5000:
                        dtw_result = compute_dtw_distance(series1, series2)
                        logger.info(
                            f"Hybrid DTW distance for {key}: {dtw_result.similarity_score:.4f}"
                        )
                        combined_metrics[key][
                            "hybrid_dtw"
                        ] = dtw_result.similarity_score
                    else:
                        logger.info(
                            f"Skipping hybrid DTW for {key} (dataset too large: {len(series1)} points)"
                        )
            else:
                logger.warning(
                    f"Hybrid key {hybrid_key} not found for distance metrics"
                )

    return combined_metrics


# =============================================================================
# Main entry point
# =============================================================================


def _emit_metrics(
    results: list,
    input_files: list,
    combined_metrics: dict,
    parsed_args,
    logger,
) -> int:
    """Log the final summary and per-key distance metrics. Returns exit code."""
    successful = len(results)
    failed = len(input_files) - successful
    logger.info(f"Analysis complete: {successful} successful, {failed} failed")

    if parsed_args.distance and combined_metrics:
        logger.info("=== Metrics Summary ===")
        for key, metrics in combined_metrics.items():
            pupitre_line = (
                f"  {key}: Euclidean={metrics.get('euclidean', 0):.4f}, "
                f"MAPE={metrics.get('mape', 0):.2f}%, "
                f"Corr={metrics.get('correlation', 0):.4f}"
                + (f", DTW={metrics['dtw']:.4f}" if "dtw" in metrics else "")
            )
            hybrid_line = (
                f"    hybrid: Euclidean={metrics.get('hybrid_euclidean', 0):.4f}, "
                f"MAPE={metrics.get('hybrid_mape', 0):.2f}%, "
                f"Corr={metrics.get('hybrid_correlation', 0):.4f}"
                + (
                    f", DTW={metrics['hybrid_dtw']:.4f}"
                    if "hybrid_dtw" in metrics
                    else ""
                )
                if "hybrid_euclidean" in metrics
                else ""
            )
            logger.info(pupitre_line)
            if hybrid_line:
                logger.info(hybrid_line)

    return 0 if failed == 0 else 1


def main(args: list[str] | None = None) -> int:
    """Main entry point for the analysis CLI.

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
    logger = _setup_logging(parsed_args)
    housing = parsed_args.housing

    try:
        config = args_to_processing_config(parsed_args)
        input_files = _collect_input_files(parsed_args, config)
        logger.info(f"Processing {len(input_files)} input files")

        if parsed_args.save:
            parsed_args.output_dir.mkdir(parents=True, exist_ok=True)

        results, all_dfs, channels_dict, pupitre_dict, hybrid_dict, keys, housing = (
            _load_records(input_files, config, parsed_args, housing, logger)
        )

        combined_metrics: dict = {}
        if not parsed_args.dry_run and results and all_dfs and keys:
            combined_metrics = _run_combined_analysis(
                results,
                all_dfs,
                keys,
                housing,
                channels_dict,
                pupitre_dict,
                hybrid_dict,
                parsed_args,
                config,
                logger,
            )

        return _emit_metrics(
            results, input_files, combined_metrics, parsed_args, logger
        )

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except (ImportError, OSError, RuntimeError) as e:
        logger.exception(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
