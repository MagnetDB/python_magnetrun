"""Info command: list keys and convert MagnetRun files."""

from __future__ import annotations

import argparse
import logging

from ..magnetdata_base import DataType

logger = logging.getLogger(__name__)


def _run(args: argparse.Namespace) -> int:
    from ..log_utils import setup_logging
    from ._shared import load_inputs

    log_level = getattr(logging, getattr(args, "log_level", "WARNING").upper(), logging.WARNING)
    setup_logging(level=log_level, log_file=getattr(args, "log_file", None))

    input_files, inputs, _extensions = load_inputs(args)
    if not inputs:
        logger.error("No files loaded.")
        return 1

    for file in input_files:
        if file not in inputs:
            continue
        mrun = inputs[file]["data"]
        mdata = mrun.getMData()

        if args.list:
            print(f"{file}: valid keys")
            mdata.info()

        if args.convert:
            import os

            f_extension = os.path.splitext(file)[-1]
            if mdata.Type == DataType.PUPITRE:
                data = mdata.getData()
                csvfile = file.replace(f_extension, ".csv")
                data.to_csv(csvfile, sep="\t", index=True, header=True)
                logger.info(f"Saved {csvfile}")
            elif mdata.Type == DataType.TDMS:
                for group in mdata.Groups:
                    df = mdata.getData(group)
                    csvfile = file.replace(f_extension, f"-{group}.csv")
                    df.to_csv(csvfile, sep="\t", index=True, header=True)
                    logger.info(f"Saved {csvfile}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    from ..cli_args import create_base_parser

    base = create_base_parser(add_input_file=False)
    p = sub.add_parser("info", parents=[base], help="list keys or convert run files")
    p.add_argument("input_file", nargs="+", help="input file(s)")
    p.add_argument("--list", action="store_true", help="list valid keys")
    p.add_argument("--convert", action="store_true", help="save to CSV")
    p.set_defaults(_handler=_run)
