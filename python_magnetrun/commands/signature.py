"""Signature command: detect regime transitions in MagnetRun data."""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def _run(args: argparse.Namespace) -> int:
    from ..log_utils import setup_logging
    from ..signature import Signature
    from ._shared import load_inputs

    log_level = getattr(logging, getattr(args, "log_level", "WARNING").upper(), logging.WARNING)
    setup_logging(level=log_level, log_file=getattr(args, "log_file", None))

    input_files, inputs, _extensions = load_inputs(args)
    if not inputs:
        logger.error("No files loaded.")
        return 1

    tkey = getattr(args, "tkey", "t")

    for file in input_files:
        if file not in inputs:
            continue
        mrun = inputs[file]["data"]
        mdata = mrun.getMData()

        try:
            sig = Signature.from_mdata(
                mdata,
                key=args.key,
                tkey=tkey,
                threshold=args.threshold,
            )
        except (KeyError, ValueError, RuntimeError) as e:
            logger.error(f"{file}: signature failed: {e}")
            continue

        logger.info(f"{file}: {sig}")

        if args.save:
            import pandas as pd

            basename = os.path.splitext(os.path.basename(file))[0]
            outfile = f"{basename}-{args.key}.csv"
            df = pd.DataFrame({"time": sig.times, args.key: sig.values})
            df.to_csv(outfile, index=False)
            print(f"Saved {outfile}")
        else:
            print(sig)

    return 0


def register(sub: argparse._SubParsersAction) -> None:
    from ..cli_args import create_base_parser

    base = create_base_parser(add_input_file=False)
    p = sub.add_parser(
        "signature",
        parents=[base],
        help="detect regime transitions for a key",
    )
    p.add_argument("input_file", nargs="+", help="input file(s)")
    p.add_argument("--key", default="Field", help="field key to analyse (default: Field)")
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="regime-detection threshold (default: 1e-3)",
    )
    p.add_argument(
        "--tkey",
        choices=["t", "timestamp"],
        default="t",
        help="time column to use (default: t)",
    )
    p.add_argument("--save", action="store_true", help="save regime CSV")
    p.set_defaults(_handler=_run)
