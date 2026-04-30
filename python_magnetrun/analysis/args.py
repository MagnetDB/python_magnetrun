"""Argument parser definitions for the analysis CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cli_args import create_base_parser, create_downsampling_parser
from .config import (
    DEFAULT_BINS,
    DEFAULT_LEVELS,
    DEFAULT_WINDOW_SIZE,
)


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the analysis CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    base_parser = create_base_parser([".tdms"])
    downsample_parser = create_downsampling_parser()
    parser = argparse.ArgumentParser(
        description="Analyze magnetrun data from TDMS and pupitre files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[base_parser, downsample_parser],
        epilog="""
Examples:
  %(prog)s M9_Overview_*.tdms --show
      Process all M9 overview files and display plots

  %(prog)s input.tdms --save --debug --log-file analysis.log
      Process with debug logging to file

    %(prog)s input.tdms --synchronize --lag --downsample-method stride --downsample-params '{"n_out": 10000}'
            Synchronize data, compute lag, and pre-downsample for plotting
        """,
    )

    # Data directories
    dir_group = parser.add_argument_group("Data directories")
    dir_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Directory for output files",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing options")
    proc_group.add_argument(
        "--tkey",
        type=str,
        choices=["t", "timestamp"],
        default="t",
        help="Time column to use for plotting",
    )
    proc_group.add_argument(
        "--synchronize",
        action="store_true",
        help="Synchronize pupitre clock with overview",
    )
    proc_group.add_argument(
        "--lag",
        action="store_true",
        help="Compute lag correlation between sources",
    )
    proc_group.add_argument(
        "--distance",
        action="store_true",
        help="Compute distance/DTW metrics between series",
    )
    proc_group.add_argument(
        "--flow-params",
        action="store_true",
        help="Compute flow parameters",
    )
    proc_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files but don't load/process data",
    )
    # Analysis parameters
    param_group = parser.add_argument_group("Analysis parameters")
    param_group.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        metavar="N",
        help="Number of bins for histograms",
    )
    param_group.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        metavar="N",
        help="Rolling window size for smoothing",
    )
    param_group.add_argument(
        "--levels",
        type=int,
        default=DEFAULT_LEVELS,
        metavar="N",
        help="Number of levels for piecewise fitting",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (requires X11)",
    )
    output_group.add_argument(
        "--save",
        action="store_true",
        help="Save plots to files (.png for matplotlib, .html for plotly backends)",
    )
    output_group.add_argument(
        "--backend",
        choices=["matplotlib", "plotly", "plotly-resampler", "plotly-widget"],
        default="matplotlib",
        help=(
            "Plotting backend. 'plotly' enables interactive HTML output. "
            "'plotly-resampler'/'plotly-widget' require a live Python kernel."
        ),
    )

    # Logging options
    log_group = parser.add_argument_group("Logging options")
    log_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (shorthand for --log-level DEBUG)",
    )
    log_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show warnings and errors",
    )
    log_group.add_argument(
        "--json-log",
        type=Path,
        metavar="FILE",
        help="Write structured JSON logs to file",
    )
    log_group.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored console output",
    )

    return parser


def args_to_downsample_config(args: argparse.Namespace):
    """
    Build a DownsampleConfig from parsed downsampling CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``downsample_method`` and optionally ``downsample_params``.

    Returns
    -------
    DownsampleConfig or None
        ``None`` when ``--downsample-method none`` (the default).

    Raises
    ------
    argparse.ArgumentTypeError
        When ``--downsample-params`` is not valid JSON.
    """
    from ..utils.downsampling import DownsampleConfig

    method = getattr(args, "downsample_method", "none")
    if method == "none":
        return None

    params_str = getattr(args, "downsample_params", None)
    params: dict = {}
    if params_str:
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"--downsample-params is not valid JSON: {exc}"
            ) from exc

    n_out: int = int(params.get("n_out", 10_000))
    bucket_size: int | None = params.get("bucket_size", None)
    return DownsampleConfig(n_out=n_out, method=method, bucket_size=bucket_size)


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    args : list of str, optional
        Arguments to parse (defaults to sys.argv[1:])

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = create_argument_parser()
    return parser.parse_args(args)


def args_to_processing_config(args: argparse.Namespace):
    """
    Convert parsed arguments to ProcessingConfig.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    ProcessingConfig
        Configuration for processing
    """
    from .processing import ProcessingConfig

    return ProcessingConfig(
        pupitre_datadir=str(args.pupitre_datadir),
        pigbrother_datadir=str(args.pigbrother_datadir),
        tkey=args.tkey,
        synchronize=args.synchronize,
        compute_lag=args.lag,
        compute_distance=args.distance,
        compute_flow_params=getattr(args, "flow_params", False),
        levels=args.levels,
        dry_run=args.dry_run,
        debug=args.debug,
        show=args.show,
        save=args.save,
        backend=getattr(args, "backend", "matplotlib"),
    )
