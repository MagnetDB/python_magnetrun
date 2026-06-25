#!/usr/bin/env python3
"""Example: inspect field metadata and groups from a pupitre or pigbrother file.

Loads a single data file (`.txt` or `.tdms`) via :func:`~python_magnetrun.MagnetRun.load_mrun`,
then prints for every field its symbol, unit, label and description, followed by
the list of fields belonging to each group, and finally the first rows of one
chosen data column.

Usage
-----
::

    # pupitre (.txt)
    python field_meta_example.py data/2025.11.05\\ -\\ 09:53:00.txt --housing M8

    # pigbrother (.tdms) — default key
    python field_meta_example.py data/M8_Overview_251105-0949.tdms --housing M8

    # explicit key
    python field_meta_example.py data/M8_Overview_251105-0949.tdms --housing M8 \\
        --key Courants_Alimentations/Champ_magn
"""

from __future__ import annotations

import argparse
import sys

from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import load_mrun

logger = get_logger(__name__)


def print_field_meta(mdata) -> None:
    """Print symbol, unit, label, and description for every field in *mdata*.

    Parameters
    ----------
    mdata : MagnetDataBase
        Loaded data object after :meth:`~python_magnetrun.magnetdata_base.MagnetDataBase.Units`
        has been called.
    """
    print(f"\n{'Field':<45} {'Symbol':<10} {'Unit':<20} {'Label':<20} Description")
    print("-" * 120)
    for key, meta in sorted(mdata.field_meta.items()):
        unit_str = f"{meta.unit:~P}" if meta.unit is not None else "—"
        print(
            f"{key:<45} {meta.symbol:<10} {unit_str:<20} {meta.label:<20} {meta.description}"
        )


def print_groups(mdata) -> None:
    """Print the fields belonging to each group defined in *mdata*.

    Parameters
    ----------
    mdata : MagnetDataBase
        Loaded data object after :meth:`~python_magnetrun.magnetdata_base.MagnetDataBase.Units`
        has been called.
    """
    groups = mdata.list_groups()
    if not groups:
        print("\n(no groups defined)")
        return

    print(f"\n{'Group':<35} Fields")
    print("-" * 80)
    for group in groups:
        raw = mdata.Groups[group]
        # pupitre: list of column names; TDMS: dict of {channel: props}
        if isinstance(raw, dict):
            channels = [ch for ch in raw if ch not in ("t", "timestamp")]
            fields = [f"{group}/{ch}" for ch in channels]
        else:
            fields = list(raw)
        print(f"{group:<35} {', '.join(fields)}")


_SKIP_KEYS = {"t", "timestamp", "Date", "Time"}


def _default_key(mdata) -> str:
    """Return the first meaningful key from *mdata.field_meta*."""
    for key in mdata.field_meta:
        if key not in _SKIP_KEYS:
            return key
    raise RuntimeError("No displayable key found in field_meta.")


def print_key_preview(mdata, key: str, n: int = 5) -> None:
    """Print metadata and the first *n* rows of the data column for *key*.

    Parameters
    ----------
    mdata : MagnetDataBase
        Loaded data object after :meth:`~python_magnetrun.magnetdata_base.MagnetDataBase.Units`
        has been called.
    key : str
        Column or ``"Group/Channel"`` key to preview.
    n : int
        Number of rows to display (default ``5``).
    """
    meta = mdata.field_meta.get(key)
    print(f"\n=== Data preview: {key!r} ===")
    if meta is not None:
        unit_str = f"{meta.unit:~P}" if meta.unit is not None else "—"
        print(f"  symbol={meta.symbol!r}  unit={unit_str}  label={meta.label!r}")
        print(f"  description: {meta.description}")
    df = mdata.getData(key)
    print(df.head(n).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", metavar="FILE", help="pupitre .txt or pigbrother .tdms file")
    parser.add_argument(
        "--housing", default="unknown", help="housing name, e.g. M8 (default: unknown)"
    )
    parser.add_argument(
        "--key", default=None, help="field key to preview (default: first meaningful key)"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.debug else "INFO")

    mrun = load_mrun(args.file, housing=args.housing)
    mdata = mrun.getMData()

    print(f"File    : {mdata.FileName}")
    print(f"Type    : {mdata.Type.name}")
    print(f"Keys    : {len(mdata.Keys)} fields")

    mdata.Units(debug=args.debug)

    print(f"\n=== Field metadata ({len(mdata.field_meta)} entries) ===")
    print_field_meta(mdata)

    print(f"\n=== Groups ({len(mdata.list_groups())} defined) ===")
    print_groups(mdata)

    key = args.key if args.key is not None else _default_key(mdata)
    print_key_preview(mdata, key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
