"""Command-line argument parsers for MagnetRun tools."""

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from .data_dirs import (
    HYBRID_DATA_DIR,
    PIGBROTHER_DATA_DIR,
    PIGBROTHER_RUNLOG_DIR,
    PUPITRE_DATA_DIR,
)


def validate_file_extension(allowed_extensions: list[str]) -> Callable[[str], str]:
    """Create a validator function for file extensions.

    :param allowed_extensions: List of allowed extensions (e.g., ['.tdms', '.txt'])
    :type allowed_extensions: list
    :return: Validator function for argparse type parameter
    :rtype: function
    """

    def validator(filepath: str) -> str:
        ext = os.path.splitext(filepath)[-1]
        if ext not in allowed_extensions:
            raise argparse.ArgumentTypeError(
                f"Invalid file extension '{ext}'. Allowed: {', '.join(allowed_extensions)}"  # noqa: E501
            )
        return filepath

    return validator


def get_datadir_mapping(args: argparse.Namespace) -> dict[str, str]:
    """Get the directory mapping for different file extensions.

    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    :return: Dictionary mapping file extensions to their data directories
    :rtype: dict
    """
    return {
        ".tdms": args.pigbrother_datadir,
        ".txt": args.pupitre_datadir,
    }


def create_common_plot_parser() -> argparse.ArgumentParser:
    """Create parser with common plotting arguments.

    :return: ArgumentParser with plotting arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--vs_time",
        help='select key(s) to plot (ex. "Field [Ucoil1]")',
        nargs="+",
        action="append",
    )
    parser.add_argument(
        "--vs_time_hybrid",
        help='select hybrid key(s) to plot (ex. "kHz/FEPC-LNCMI/I_H1")',
        nargs="+",
    )
    parser.add_argument(
        "--key_vs_key",
        help='select pair(s) of keys to plot (ex. "Field-Icoil1")',
        nargs="+",
        action="append",
    )
    parser.add_argument(
        "--normalize", help="normalize data before plot", action="store_true"
    )
    parser.add_argument(
        "--backend",
        choices=["matplotlib", "plotly", "plotly-resampler", "plotly-widget"],
        default="matplotlib",
        help=(
            "plotting backend. 'plotly' enables interactive HTML output and JSON export. "
            "'plotly-resampler'/'plotly-widget' require a live Python kernel."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print Plotly JSON spec to stdout instead of displaying/saving (requires --backend plotly)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--subplots",
        action="store_true",
        default=False,
        help="display each field in a separate subplot sharing the time axis (default: overlay on one axes)",
    )
    mode_group.add_argument(
        "--overlay",
        action="store_true",
        default=False,
        help="overlay all fields on one axes (default mode; explicit flag for clarity)",
    )
    parser.add_argument(
        "--unit",
        metavar="FIELD=UNIT",
        action="append",
        default=None,
        help=(
            "override display unit for a field (pint-parseable); repeatable. "
            "Example: --unit Field_B=tesla --unit Courant_GR1=ampere"
        ),
    )
    parser.add_argument(
        "--same-color-per-type",
        action="store_true",
        default=False,
        help=(
            "use one fixed color per file type (.txt → pupitre, .tdms → archive) "
            "instead of a distinct palette color per file"
        ),
    )
    parser.add_argument(
        "--plot-config",
        metavar="FILE",
        type=Path,
        default=None,
        help="path to a plot config JSON file (colors + style); overrides built-in defaults",
    )
    parser.add_argument(
        "--marker",
        metavar="MARKER",
        default=None,
        help=(
            "matplotlib marker style for key_vs_key plots (e.g. 'o', '+', 'x', 's'). "
            "Default: no marker."
        ),
    )
    parser.add_argument(
        "--no-lines",
        action="store_true",
        default=False,
        help="for key_vs_key: draw only markers, suppress connecting lines (implies --marker o if no --marker given)",
    )
    parser.add_argument(
        "--field_style",
        metavar="FIELD=STYLESPEC",
        action="append",
        default=None,
        help=(
            "per-field plot style for --vs_time; repeatable. "
            "STYLESPEC syntax: [LINESTYLE][MARKER][:N] where LINESTYLE is '-', '--' or '-.', "
            "MARKER is any matplotlib marker (e.g. 'o', '+', 's'), and N is markevery. "
            "Examples: --field_style Référence_A1=- "
            "--field_style Référence_A2=o:10 "
            "--field_style Référence_A2=-o:5"
        ),
    )
    return parser


def create_datadir_parser() -> argparse.ArgumentParser:
    """Create parser with data directory arguments (pupitre, pigbrother).

    Env var resolution order:
    - pupitre:    MAGNETRUN_PUPITRE_DATA_DIR > PUPITRE_DATADIR
    - pigbrother: MAGNETRUN_PIGBROTHER_DATA_DIR > PIGBROTHER_DATADIR

    :return: ArgumentParser with data directory arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pupitre_datadir",
        help=(
            "root directory for pupitre .txt files "
            "(overrides MAGNETRUN_PUPITRE_DATA_DIR / PUPITRE_DATADIR )"
        ),
        type=str,
        default=PUPITRE_DATA_DIR,
    )
    parser.add_argument(
        "--pigbrother_datadir",
        help=("root directory for pigbrother .tdms files "),
        type=str,
        default=PIGBROTHER_DATA_DIR,
    )
    return parser


def create_hybrid_parser() -> argparse.ArgumentParser:
    """Create parser with hybrid data arguments.

    :return: ArgumentParser with hybrid data arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--hybrid_datadir",
        help=(
            "base directory for hybrid (kHz/rms/trigger) data "
            "(overrides MAGNETRUN_HYBRID_DATA_DIR / HYBRID_DATADIR)"
        ),
        type=str,
        default=HYBRID_DATA_DIR or None,
    )
    parser.add_argument(
        "--hybrid_date",
        help="date for hybrid data in YYYY-MM-DD format",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fepc_system",
        help="FEPC system name",
        type=str,
        choices=["FEPC-LNCMI", "FEPC-AUX-LNCMI"],
        default="FEPC-LNCMI",
    )
    parser.add_argument(
        "--hybrid_downsample",
        help="downsample kHz data to N points for plotting",
        type=int,
        default=50000,
    )
    return parser


DOWNSAMPLE_METHODS = ("none", "stride", "minmax", "minmax_lttb", "lttb")
"""Valid downsampling method names for ``--downsample-method``."""


def create_downsampling_parser() -> argparse.ArgumentParser:
    """Create parser with downsampling arguments.

    ``--downsample-method`` selects the algorithm (``none`` disables all
    downsampling, which is the default).  ``--downsample-params`` accepts a
    JSON object whose keys are forwarded as keyword arguments to
    :class:`~python_magnetrun.utils.downsampling.DownsampleConfig`.

    Supported ``--downsample-params`` keys:

    * ``n_out`` *(int)*: target number of output points (default: 10 000).
    * ``bucket_size`` *(int)*: only used by the ``minmax`` method.

    Examples::

        --downsample-method stride --downsample-params '{"n_out": 5000}'
        --downsample-method minmax_lttb
        --downsample-method minmax --downsample-params '{"n_out": 8000, "bucket_size": 50}'

    :return: ArgumentParser with downsampling arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--downsample-method",
        choices=DOWNSAMPLE_METHODS,
        default="none",
        metavar="METHOD",
        help=(
            "downsampling algorithm applied before plotting: "
            "none (disabled, default), stride, minmax, minmax_lttb, lttb. "
            "minmax_lttb and lttb require the tsdownsample package."
        ),
    )
    parser.add_argument(
        "--downsample-params",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "JSON object of method-specific parameters. "
            "Supported keys: n_out (int, default 10000), "
            "bucket_size (int, minmax only). "
            "Example: '{\"n_out\": 5000}'"
        ),
    )
    return parser


def create_common_smoothing_parser() -> argparse.ArgumentParser:
    """Create parser with common smoothing arguments.

    :return: ArgumentParser with smoothing arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--smoother",
        help="smooth selected data with selected methods: [ag, bell_kernel, statsmodel_sm, savgol]",  # noqa: E501
        type=str,
        choices=["ag", "bell_kernel", "statsmodel_sm", "savgol"],
    )
    parser.add_argument("--window", help="size of rolling window", type=int, default=10)
    parser.add_argument(
        "--smoothing_f", help="set smoothing_f", type=float, default=0.7
    )
    parser.add_argument(
        "--smoothing_tau", help="set smoothing_tau", type=float, default=400
    )
    parser.add_argument(
        "--smoothing_iter", help="set smoothing_iter", type=int, default=3
    )
    return parser


def create_base_parser(
    allowed_extensions: list[str] | None = None,
    add_input_file: bool = True,
) -> argparse.ArgumentParser:
    """Create base parser with common arguments.

    Includes data directory arguments via :func:`create_datadir_parser`.

    :param allowed_extensions: Optional list of allowed file extensions (e.g., ['.tdms', '.txt'])  # noqa: E501
    :type allowed_extensions: list or None
    :param add_input_file: Whether to add the positional ``input_file`` argument (default: True).
        Pass ``False`` when the script does not take input files.
    :type add_input_file: bool
    :return: ArgumentParser with base arguments
    :rtype: argparse.ArgumentParser
    """
    datadir_parser = create_datadir_parser()
    parser = argparse.ArgumentParser(add_help=False, parents=[datadir_parser])

    # Add input_file with optional extension validation
    if add_input_file:
        if allowed_extensions:
            parser.add_argument(
                "input_file",
                nargs="+",
                type=validate_file_extension(allowed_extensions),
                help=f"enter input file (allowed: {', '.join(allowed_extensions)})",
            )
        else:
            parser.add_argument("input_file", nargs="+", help="enter input file")

    parser.add_argument(
        "--site",
        help="specify a site (ex. M9_M25032101_0)",
        default="notdefined",
    )
    parser.add_argument(
        "--insert", help="specify an insert (ex: M25032101)", default="notdefined"
    )
    parser.add_argument(
        "--housing", help="specify a housing (ex. M8, M9,...)", default="notdefined"
    )
    parser.add_argument(
        "--log-level",
        help="set logging level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--log-file",
        help="path to log file (if not specified, logs to console)",
        type=Path,
        default=None,
    )
    return parser


def create_managed_plots_parser() -> argparse.ArgumentParser:
    """Create parser with managed plot options (save and show).

    :return: ArgumentParser with save and show arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help=(
            "save figure to FILE (.png/.html chosen by backend); "
            "omit FILE for an auto-generated name in the current directory"
        ),
    )
    group.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="display figure interactively (default when neither flag is given)",
    )
    return parser


def create_analysis_parser() -> argparse.ArgumentParser:
    """Create the argument parser for analysis-refactor.

    :return: Configured ArgumentParser for analysis
    :rtype: argparse.ArgumentParser
    """
    base_parser = create_base_parser()
    managed_plots_parser = create_managed_plots_parser()

    parser = argparse.ArgumentParser(parents=[base_parser, managed_plots_parser])
    parser.add_argument(
        "--runlogs", nargs="+", help="enter run-log files from ACQ_ENET"
    )
    parser.add_argument(
        "--runlog_datadir",
        help=(
            "root directory for ACQ_ENET run-log files "
            "(overrides MAGNETRUN_PIGBROTHER_RUNLOG_DIR)"
        ),
        type=str,
        default=PIGBROTHER_RUNLOG_DIR or None,
    )
    parser.add_argument(
        "--tkey",
        help="choose tkey",
        choices=["t", "timestamp"],
        type=str,
        default="t",
    )
    parser.add_argument("--dry_run", help="dry_run mode", action="store_true")
    parser.add_argument(
        "--synchronize",
        help="synchronize clock pupitre/pigbrother files",
        action="store_true",
    )
    parser.add_argument(
        "--flow",
        help="compute flow params from pupitre",
        action="store_true",
    )
    parser.add_argument(
        "--lag",
        help="compute lag between pupitre and pigbrother data",
        action="store_true",
    )
    parser.add_argument(
        "--distance", help="compute distance between series", action="store_true"
    )
    parser.add_argument("--bins", help="set bins for histograms", type=int, default=10)
    parser.add_argument(
        "--window", help="set rolling window size", type=int, default=50
    )
    parser.add_argument("--levels", help="set levels", type=int, default=4)
    parser.add_argument(
        "--plot-percent",
        help="percentage of points to plot (0-100, default 10)",
        type=float,
        default=10.0,
    )

    return parser
