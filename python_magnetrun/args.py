"""Argument parser definitions for the python_magnetrun CLI."""

import argparse

from .cli_args import (
    create_base_parser,
    create_common_plot_parser,
    create_common_smoothing_parser,
    create_hybrid_parser,
    create_managed_plots_parser,
    get_datadir_mapping,
)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for python_magnetrun.

    :return: Configured ArgumentParser with all subcommands
    :rtype: argparse.ArgumentParser
    """
    base_parser = create_base_parser()
    plot_parser = create_common_plot_parser()
    hybrid_parser = create_hybrid_parser()
    smoothing_parser = create_common_smoothing_parser()
    managed_plots_parser = create_managed_plots_parser()

    parser = argparse.ArgumentParser(parents=[base_parser, hybrid_parser])

    subparsers = parser.add_subparsers(title="commands", dest="command", help="sub-command help")

    # Info subcommand
    parser_info = subparsers.add_parser("info", help="info help")
    parser_info.add_argument("--list", help="list key in csv", action="store_true")
    parser_info.add_argument("--convert", help="save to csv", action="store_true")

    # Add subcommand (with plot capabilities)
    parser_add = subparsers.add_parser(
        "add",
        help="add help",
        parents=[plot_parser, managed_plots_parser],
    )
    parser_add.add_argument(
        "--formula", help="add new column with associated formula", type=str, default=""
    )
    parser_add.add_argument("--symbol", help="physical symbol for the new field (e.g. 'B', 'I')", type=str, default="")
    parser_add.add_argument("--unit", help="unit string for the new field (e.g. 'tesla', 'ampere')", type=str, default="")
    parser_add.add_argument("--label", help="human-readable label for the new field", type=str, default="")
    parser_add.add_argument("--description", help="description of the new field", type=str, default="")
    parser_add.add_argument("--compute", help="compute", action="store_true")
    parser_add.add_argument("--plot", help="plot", action="store_true")

    # Plot subcommand
    _parser_plot = subparsers.add_parser(
        "plot",
        help="plot help",
        parents=[plot_parser, managed_plots_parser],
    )

    # Select subcommand
    parser_select = subparsers.add_parser("select", help="select help", parents=[smoothing_parser])
    parser_select.add_argument("--output_time", nargs="+", help="output key(s) for time")
    parser_select.add_argument(
        "--output_timerange",
        help="set time range to extract (YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS, local time)",
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
    parser_select.add_argument("--convert", help="convert file to csv", action="store_true")

    # Stats subcommand
    parser_stats = subparsers.add_parser("stats", help="stats help", parents=[managed_plots_parser])
    parser_stats.add_argument("--detect_bkpts", help="find breaking points", action="store_true")
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
    parser_stats.add_argument("--window", help="size of rolling window", type=int, default=10)
    parser_stats.add_argument("--level", help="select level", type=int, default=90)

    return parser


__all__ = ["create_argument_parser", "get_datadir_mapping"]
