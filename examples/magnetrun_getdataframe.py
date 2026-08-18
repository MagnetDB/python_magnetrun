#!/usr/bin/env python3
"""
Example: using MagnetRun.getDataFrame()

Demonstrates loading pupitre (.txt) and pigbrother/TDMS (.tdms) files and
retrieving their contents as pandas DataFrame(s) via getDataFrame().

Usage:
    python magnetrun_getdataframe.py --pupitre M10_run.txt --tdms M10_overview.tdms
    python magnetrun_getdataframe.py --pupitre M10_run.txt
    python magnetrun_getdataframe.py --tdms M10_overview.tdms
"""

import argparse
import sys

from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import MagnetRun

logger = get_logger(__name__)


def demo_pupitre(filepath: str, housing: str = "M10") -> None:
    print(f"\n=== Pupitre: {filepath} ===")
    mrun = MagnetRun.fromtxt(housing, "Unknown", filepath)

    result = mrun.getDataFrame()

    # For pupitre data getDataFrame() returns a single DataFrame
    import pandas as pd
    assert isinstance(result, pd.DataFrame), "Expected a single DataFrame for pupitre"

    print(f"Shape      : {result.shape}")
    print(f"Columns    : {list(result.columns)}")
    print(f"Start time : {mrun.StartTime}")
    print(result.head())


def demo_tdms(filepath: str, housing: str = "M10") -> None:
    print(f"\n=== TDMS (pigbrother): {filepath} ===")
    mrun = MagnetRun.fromtdms(housing, "Unknown", filepath)

    result = mrun.getDataFrame()

    # For TDMS data getDataFrame() returns a list of DataFrames, one per group
    assert isinstance(result, list), "Expected a list of DataFrames for TDMS"

    mdata = mrun.getMData()
    groups = list(mdata.Groups.keys())
    print(f"Number of groups: {len(result)}")
    for group, df in zip(groups, result, strict=False):
        print(f"\n  Group '{group}': shape={df.shape}, columns={list(df.columns)}")
        print(df.head(3))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pupitre", metavar="FILE", help="pupitre .txt file")
    parser.add_argument("--tdms", metavar="FILE", help="pigbrother .tdms file")
    parser.add_argument("--housing", default="M10", help="housing name (default: M10)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.debug else "INFO")

    if not args.pupitre and not args.tdms:
        parser.error("provide at least one of --pupitre or --tdms")

    if args.pupitre:
        demo_pupitre(args.pupitre, housing=args.housing)

    if args.tdms:
        demo_tdms(args.tdms, housing=args.housing)

    return 0


if __name__ == "__main__":
    sys.exit(main())
