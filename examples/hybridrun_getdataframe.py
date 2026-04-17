#!/usr/bin/env python3
"""
Example: using HybridRun.getDataFrame()

Demonstrates loading hybrid kHz/RMS data and retrieving all available
variables as a list of pandas DataFrames via getDataFrame().

Each DataFrame in the returned list corresponds to one key from
HybridRun.getKeys() and has columns ['time', <key>].

Usage:
    python hybridrun_getdataframe.py --date 2025-11-05 --system FEPC-AUX-LNCMI
    python hybridrun_getdataframe.py --date 2025-11-05 --system FEPC-LNCMI --hours 13 14 15
    python hybridrun_getdataframe.py --date 2025-11-05 --system FEPC-AUX-LNCMI --key kHz/FEPC-AUX-LNCMI/ALIM1_J1
"""

import argparse
import sys

from python_magnetrun.data_dirs import HYBRID_DATA_DIR
from python_magnetrun.hybrid.hybrid_run import HybridRun
from python_magnetrun.log_utils import get_logger, setup_logging

logger = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=HYBRID_DATA_DIR, help="hybrid base directory")
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD", help="date to load")
    parser.add_argument("--system", default=None, metavar="FEPC_SYSTEM", help="FEPC system name")
    parser.add_argument("--housing", default="Hybrid", help="housing name")
    parser.add_argument("--hours", nargs="*", type=int, default=None, help="hours to load")
    parser.add_argument(
        "--key",
        default=None,
        metavar="KEY",
        help="show only DataFrames whose key contains this string",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.debug else "INFO")

    print(f"Loading HybridRun: base_dir={args.base_dir}, date={args.date}, system={args.system}")

    hrun = HybridRun.fromdir(
        base_dir=args.base_dir,
        date_str=args.date,
        fepc_system=args.system,
        housing=args.housing,
    )

    print(f"Available keys ({len(hrun.getKeys())}): {hrun.getKeys()[:10]} ...")

    # Retrieve all available variables as a list of DataFrames
    frames = hrun.getDataFrame()

    print(f"\ngetDataFrame() returned {len(frames)} DataFrames")

    # Filter by key substring if requested
    all_keys = hrun.getKeys()
    filtered = [
        (k, df)
        for k, df in zip(all_keys, frames, strict=False)
        if args.key is None or args.key in k
    ]

    if not filtered:
        print(f"No DataFrames match key filter '{args.key}'")
        return 1

    for key, df in filtered:
        print(f"\n  key='{key}': shape={df.shape}, columns={list(df.columns)}")
        print(df.head(3).to_string(index=False))
        print(f"  time range: {df['time'].min():.1f} – {df['time'].max():.1f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
