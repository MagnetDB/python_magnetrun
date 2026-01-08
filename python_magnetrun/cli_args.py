"""Command-line argument parsers for MagnetRun tools."""

import argparse
import os


def validate_file_extension(allowed_extensions):
    """Create a validator function for file extensions.

    :param allowed_extensions: List of allowed extensions (e.g., ['.tdms', '.txt'])
    :type allowed_extensions: list
    :return: Validator function for argparse type parameter
    :rtype: function
    """

    def validator(filepath):
        ext = os.path.splitext(filepath)[-1]
        if ext not in allowed_extensions:
            raise argparse.ArgumentTypeError(
                f"Invalid file extension '{ext}'. Allowed: {', '.join(allowed_extensions)}"
            )
        return filepath

    return validator


def get_datadir_mapping(args):
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


def create_common_plot_parser():
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
        "--key_vs_key",
        help='select pair(s) of keys to plot (ex. "Field-Icoil1")',
        nargs="+",
        action="append",
    )
    parser.add_argument(
        "--normalize", help="normalize data before plot", action="store_true"
    )
    return parser


def create_common_smoothing_parser():
    """Create parser with common smoothing arguments.

    :return: ArgumentParser with smoothing arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--smoother",
        help="smooth selected data with selected methods: [ag, bell_kernel, statsmodel_sm, savgol]",
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


def create_base_parser(allowed_extensions=None):
    """Create base parser with common arguments.

    :param allowed_extensions: Optional list of allowed file extensions (e.g., ['.tdms', '.txt'])
    :type allowed_extensions: list or None
    :return: ArgumentParser with base arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)

    # Add input_file with optional extension validation
    if allowed_extensions:
        parser.add_argument(
            "input_file",
            nargs="+",
            type=validate_file_extension(allowed_extensions),
            help=f"enter input file (allowed: {', '.join(allowed_extensions)})",
        )
    else:
        parser.add_argument("input_file", nargs="+", help="enter input file")

    parser.add_argument("--site", help="specify a site (ex. M8, M9,...)", default="M9")
    parser.add_argument("--insert", help="specify an insert", default="notdefined")
    parser.add_argument(
        "--pupitre_datadir",
        help="enter pupitre datadir (default srvdata)",
        type=str,
        default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install",
    )
    parser.add_argument(
        "--pigbrother_datadir",
        help="enter pigbrother datadir (default pigbrotherdata)",
        type=str,
        default="/home/LNCMI-G/christophe.trophime/github/python_magnetrun/pigbrotherdata/Fichiers_Data",
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
        type=str,
        default=None,
    )
    return parser


def create_managed_plots_parser():
    """Create parser with managed plot options (save and show).

    :return: ArgumentParser with save and show arguments
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--save", help="save graphs (png format)", action="store_true")
    parser.add_argument(
        "--show", help="display graphs (require X11)", action="store_true"
    )
    return parser


def create_main_parser():
    """Create the main argument parser for python_magnetrun.

    :return: Configured ArgumentParser with all subcommands
    :rtype: argparse.ArgumentParser
    """
    base_parser = create_base_parser()
    plot_parser = create_common_plot_parser()
    smoothing_parser = create_common_smoothing_parser()
    managed_plots_parser = create_managed_plots_parser()

    parser = argparse.ArgumentParser(parents=[base_parser])

    subparsers = parser.add_subparsers(
        title="commands", dest="command", help="sub-command help"
    )

    # Info subcommand
    parser_info = subparsers.add_parser("info", help="info help")
    parser_info.add_argument("--list", help="list key in csv", action="store_true")
    parser_info.add_argument("--convert", help="save to csv", action="store_true")

    # Add subcommand (with plot capabilities)
    parser_add = subparsers.add_parser(
        "add", help="add help", parents=[plot_parser, managed_plots_parser]
    )
    parser_add.add_argument(
        "--formula", help="add new column with associated formula", type=str, default=""
    )
    parser_add.add_argument("--compute", help="compute", action="store_true")
    parser_add.add_argument("--plot", help="plot", action="store_true")

    # Plot subcommand
    parser_plot = subparsers.add_parser(
        "plot", help="plot help", parents=[plot_parser, managed_plots_parser]
    )

    # Select subcommand
    parser_select = subparsers.add_parser(
        "select", help="select help", parents=[smoothing_parser]
    )
    parser_select.add_argument(
        "--output_time", nargs="+", help="output key(s) for time"
    )
    parser_select.add_argument(
        "--output_timerange",
        help="set time range to extract (start;end)",
        action="append",
    )
    parser_select.add_argument(
        "--output_key",
        nargs="+",
        help="output key(s) for time",
        action="append",
    )
    parser_select.add_argument(
        "--extract_pairkeys",
        nargs="+",
        help="dump key(s) to file",
        action="append",
    )
    parser_select.add_argument(
        "--convert", help="convert file to csv", action="store_true"
    )

    # Stats subcommand
    parser_stats = subparsers.add_parser(
        "stats", help="stats help", parents=[managed_plots_parser]
    )
    parser_stats.add_argument(
        "--detect_bkpts", help="find breaking points", action="store_true"
    )
    parser_stats.add_argument("--localmax", help="find local max", action="store_true")
    parser_stats.add_argument("--plateau", help="find plateau", action="store_true")
    parser_stats.add_argument(
        "--keys",
        help="select key(s) to perform selected stats",
        nargs="+",
    )
    parser_stats.add_argument(
        "--threshold",
        help="specify threshold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser_stats.add_argument(
        "--bthreshold",
        help="specify b threshold for regime detection",
        type=float,
        default=1.0e-3,
    )
    parser_stats.add_argument(
        "--dthreshold",
        help="specify duration threshold for regime detection",
        type=float,
        default=10,
    )
    parser_stats.add_argument(
        "--window", help="size of rolling window", type=int, default=10
    )
    parser_stats.add_argument("--level", help="select level", type=int, default=90)

    return parser


def create_analysis_parser():
    """Create the argument parser for analysis-refactor.

    :return: Configured ArgumentParser for analysis
    :rtype: argparse.ArgumentParser
    """
    base_parser = create_base_parser()
    managed_plots_parser = create_managed_plots_parser()

    parser = argparse.ArgumentParser(parents=[base_parser, managed_plots_parser])
    parser.add_argument("--logs", nargs="+", help="enter log files from ACQ_ENET")
    parser.add_argument(
        "--log_datadir",
        help="enter log datadir (default srvdata)",
        type=str,
        default="/home/LNCMI-G/christophe.trophime/LNCMIG-Data/srv-data-install",
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
