"""
pupitre_to_duckdb.py
====================
Extract data from pupitre ``.txt`` files and store in a DuckDB database.

Each housing (M9, M10, …) is stored in its own table.  The housing name is
auto-detected from the parent directory of each input file; use ``--housing``
to override.  Rows with duplicate timestamps are silently skipped.

Usage
-----
    python pupitre_to_duckdb.py \\
        --fields Field Icoil1 Ucoil1 Pmagnet \\
        --output pupitre.duckdb \\
        "srv-data-install/M9/*.txt"

    # Explicit housing when directory name is not the magnet name
    python pupitre_to_duckdb.py \\
        --fields Field Pmagnet \\
        --housing M9 \\
        --output pupitre.duckdb \\
        /tmp/flat_dir/*.txt
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import duckdb
import pandas as pd
from natsort import natsorted

from python_magnetrun.cli_args import create_base_parser
from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import MagnetRun

logger = get_logger(__name__)


def _expand_patterns(patterns: list[str]) -> list[str]:
    """Expand glob patterns into a natsorted file list.

    Parameters
    ----------
    patterns : list[str]
        File paths or glob patterns (e.g. ``'M9/*.txt'``).

    Returns
    -------
    list[str]
        Natsorted list of matched file paths.

    Raises
    ------
    SystemExit
        If no files match any of the provided patterns.
    """
    matched: list[str] = []
    for pattern in patterns:
        hits = glob.glob(pattern)
        if not hits:
            logger.warning("No files matched pattern: %s", pattern)
        matched.extend(hits)
    if not matched:
        print("Error: no files matched the provided patterns.", file=sys.stderr)
        sys.exit(1)
    return natsorted(matched)


_HOUSING_UNSET = "notdefined"


def _detect_housing(filepath: str, housing_override: str | None) -> str:
    """Return the housing name for *filepath*.

    Parameters
    ----------
    filepath : str
        Path to the pupitre ``.txt`` file.
    housing_override : str, optional
        Explicit housing name supplied via ``--housing``; always used when set
        to a value other than ``'notdefined'``.

    Returns
    -------
    str
        Housing name (e.g. ``'M9'``, ``'M10'``).

    Raises
    ------
    ValueError
        When the housing cannot be determined from the parent directory and no
        override was given.
    """
    if housing_override and housing_override != _HOUSING_UNSET:
        return housing_override
    parent = Path(filepath).parent.name
    if parent and parent not in (".", ""):
        return parent
    raise ValueError(
        f"Cannot determine housing from parent directory '{parent}' for "
        f"'{filepath}'. Use --housing to specify it explicitly."
    )


def _ensure_table(
    con: duckdb.DuckDBPyConnection, housing: str, fields: list[str]
) -> None:
    """Create the housing table if it does not already exist.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    housing : str
        Table name (magnet housing identifier, e.g. ``'M9'``).
    fields : list[str]
        Data column names (excluding ``timestamp``); stored as ``DOUBLE``.
    """
    col_defs = ", ".join(f'"{f}" DOUBLE' for f in fields)
    con.execute(
        f'CREATE TABLE IF NOT EXISTS "{housing}" '
        f"(timestamp TIMESTAMP NOT NULL PRIMARY KEY, {col_defs})"
    )


def _insert_dataframe(
    con: duckdb.DuckDBPyConnection,
    housing: str,
    df: pd.DataFrame,
    fields: list[str],
) -> int:
    """Insert rows from *df* into the housing table, skipping duplicates.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Open DuckDB connection.
    housing : str
        Target table name.
    df : pandas.DataFrame
        Source data; must contain a ``timestamp`` column.
    fields : list[str]
        Requested data column names.  Columns absent from *df* are inserted as
        ``NULL``.

    Returns
    -------
    int
        Number of rows actually inserted (duplicates excluded).
    """
    select_parts = ["timestamp"]
    for f in fields:
        if f in df.columns:
            select_parts.append(f'"{f}"')
        else:
            select_parts.append(f'NULL::DOUBLE AS "{f}"')

    select_sql = ", ".join(select_parts)

    before = con.execute(f'SELECT COUNT(*) FROM "{housing}"').fetchone()[0]
    con.register("_tmp_pupitre", df)
    con.execute(
        f'INSERT OR IGNORE INTO "{housing}" '
        f"SELECT {select_sql} FROM _tmp_pupitre"
    )
    con.unregister("_tmp_pupitre")
    after = con.execute(f'SELECT COUNT(*) FROM "{housing}"').fetchone()[0]
    return after - before


def main() -> None:
    """Entry point for the pupitre-to-DuckDB extraction tool."""
    base_parser = create_base_parser(add_input_file=False)
    parser = argparse.ArgumentParser(
        parents=[base_parser],
        description="Extract pupitre .txt data and store in a DuckDB database.",
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        metavar="PATTERN",
        help="Pupitre .txt file paths or glob patterns (e.g. 'M9/*.txt').",
    )
    parser.add_argument(
        "--output",
        default="pupitre.duckdb",
        metavar="FILE",
        help="DuckDB output file (default: pupitre.duckdb).",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        required=True,
        metavar="FIELD",
        help="Data columns to extract; timestamp is always included.",
    )
    args = parser.parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.debug("args: %s", args)

    files = _expand_patterns(args.input_files)
    print(f"Processing {len(files)} file(s)...")

    # Resolve housing for every file before opening DuckDB (fail fast on errors)
    file_housing: dict[str, str] = {}
    for f in files:
        try:
            file_housing[f] = _detect_housing(f, args.housing)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    con = duckdb.connect(args.output)

    # Create tables for all unique housings
    for housing in sorted(set(file_housing.values())):
        _ensure_table(con, housing, args.fields)

    # Process files
    totals: dict[str, int] = {}
    for filepath in files:
        housing = file_housing[filepath]
        print(f"  {filepath} → {housing}", end="", flush=True)
        try:
            mrun = MagnetRun.fromtxt(housing, "", filepath)
            df = mrun.getDataFrame()
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"  [SKIP: {exc}]")
            logger.warning("Failed to load %s: %s", filepath, exc)
            continue

        missing = [f for f in args.fields if f not in df.columns]
        if missing:
            logger.warning(
                "%s: field(s) not found and will be NULL: %s",
                Path(filepath).name,
                ", ".join(missing),
            )

        n = _insert_dataframe(con, housing, df, args.fields)
        totals[housing] = totals.get(housing, 0) + n
        print(f"  +{n} rows")

    con.close()

    print(f"\nDone → {args.output}")
    for housing, count in sorted(totals.items()):
        print(f"  {housing}: {count} row(s) inserted")


if __name__ == "__main__":
    main()
