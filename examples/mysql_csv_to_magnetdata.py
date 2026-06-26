#!/usr/bin/env python3
"""
mysql_csv_to_magnetdata.py
==========================
Demonstrate loading a CSV exported by ``mysql_connect.py --mode export``
into a :class:`~python_magnetrun.magnetdata_pandas.PandasMagnetData` object.

DuckDB's CSV export (``COPY … TO … (HEADER, DELIMITER ',')``) produces a
plain comma-separated file with a single header row and no skip rows — the
exact format expected by :class:`~python_magnetrun.readers.csv_readers.CsvReader`.

Two modes
---------
generate
    Write a synthetic CSV that matches the DuckDB export format (comma-
    separated, ISO 8601 timestamps), then load and inspect it.  Use this
    mode to test the round-trip without a live MySQL server.

load
    Load an existing CSV (produced by ``mysql_connect.py --mode export``)
    and inspect it.

Usage
-----
    # Generate a synthetic CSV and load it
    python mysql_csv_to_magnetdata.py generate --output sample.csv [--plot]

    # Load an existing exported CSV
    python mysql_csv_to_magnetdata.py load sample.csv [--timestamp-col timestamp] [--plot]

    # Load and plot specific columns
    python mysql_csv_to_magnetdata.py load measurements.csv \\
        --timestamp-col timestamp --fields Icoil Ucoil --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def generate_sample_csv(output: Path, n_rows: int = 300) -> None:
    """Write a synthetic CSV that mimics a DuckDB mysql_scanner export.

    Parameters
    ----------
    output : Path
        Destination ``.csv`` file.
    n_rows : int
        Number of data rows to generate.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    t = pd.date_range("2024-03-15 08:00:00", periods=n_rows, freq="10s")

    # Mimic a measurements table: timestamp + two current channels + a voltage
    df = pd.DataFrame(
        {
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),  # DuckDB TIMESTAMP → string
            "Icoil": 1000.0 + rng.normal(0, 5, n_rows),
            "Ucoil": 12.5 + rng.normal(0, 0.2, n_rows),
            "teb": 15.0 + rng.normal(0, 0.3, n_rows),
        }
    )

    # DuckDB uses HEADER + DELIMITER ',' — replicate exactly
    df.to_csv(output, index=False, sep=",")
    print(f"Generated {n_rows}-row sample CSV → {output}")


# ---------------------------------------------------------------------------
# Load and inspect
# ---------------------------------------------------------------------------


def load_and_inspect(
    csv_path: Path,
    timestamp_col: str | None,
    fields: list[str] | None,
) -> pd.DataFrame:
    """Load *csv_path* as a PandasMagnetData and return the requested DataFrame.

    Parameters
    ----------
    csv_path : Path
        CSV file produced by ``mysql_connect.py --mode export``.
    timestamp_col : str or None
        Name of the TIMESTAMP column; when given it is parsed to
        ``datetime64`` (DuckDB serialises timestamps as ISO 8601 strings).
    fields : list[str] or None
        Columns to return; ``None`` returns all columns.

    Returns
    -------
    pandas.DataFrame
        Requested data with the timestamp column already parsed.
    """
    from python_magnetrun.magnetdata_pandas import PandasMagnetData

    mdata = PandasMagnetData.fromcsv(str(csv_path))

    print(f"\n--- MagnetData loaded from {csv_path.name} ---")
    print(f"Keys  : {mdata.getKeys()}")

    df = mdata.getData(fields)
    print(f"Shape : {df.shape}")
    print(f"dtypes:\n{df.dtypes.to_string()}")

    # DuckDB serialises TIMESTAMP as ISO 8601 strings; parse them explicitly.
    if timestamp_col and timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        print(f"\nTimestamp column '{timestamp_col}' parsed to {df[timestamp_col].dtype}")
        print(f"Range : {df[timestamp_col].iloc[0]}  →  {df[timestamp_col].iloc[-1]}")

    print(f"\nFirst rows:\n{df.head(3).to_string(index=False)}")
    return df


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------


def plot_data(df: pd.DataFrame, timestamp_col: str | None, fields: list[str]) -> None:
    """Plot *fields* against the timestamp (or row index).

    Parameters
    ----------
    df : pandas.DataFrame
        Data returned by :func:`load_and_inspect`.
    timestamp_col : str or None
        Column to use as x-axis; falls back to row index when ``None``.
    fields : list[str]
        Numeric columns to plot on the y-axis.
    """
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.  pip install matplotlib")
        return

    x = df[timestamp_col] if timestamp_col and timestamp_col in df.columns else df.index
    is_datetime = pd.api.types.is_datetime64_any_dtype(x)

    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 3 * len(fields)),
                              sharex=True, squeeze=False)

    for ax, field in zip(axes[:, 0], fields):
        if field not in df.columns:
            print(f"Warning: column '{field}' not found — skipping")
            continue
        ax.plot(x, df[field], linewidth=0.9)
        ax.set_ylabel(field)
        ax.grid(True, linestyle="--", alpha=0.5)

    if is_datetime:
        locator = mdates.AutoDateLocator()
        axes[-1, 0].xaxis.set_major_locator(locator)
        axes[-1, 0].xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
        fig.autofmt_xdate()

    xlabel = timestamp_col if timestamp_col else "row index"
    axes[-1, 0].set_xlabel(xlabel)
    fig.suptitle(f"MySQL CSV export → MagnetData  ({', '.join(fields)})")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # generate subcommand
    gen = sub.add_parser(
        "generate",
        help="write a synthetic DuckDB-style CSV then load it",
    )
    gen.add_argument(
        "--output", metavar="FILE", default="sample_mysql_export.csv",
        help="path of the CSV to create (default: sample_mysql_export.csv)",
    )
    gen.add_argument(
        "--rows", type=int, default=300, metavar="N",
        help="number of data rows to generate (default: 300)",
    )
    gen.add_argument("--plot", action="store_true", help="plot after loading")

    # load subcommand
    ld = sub.add_parser(
        "load",
        help="load an existing CSV exported by mysql_connect.py",
    )
    ld.add_argument("csv", metavar="FILE", help="CSV file to load")
    ld.add_argument(
        "--timestamp-col", metavar="COL", default=None, dest="timestamp_col",
        help="TIMESTAMP column name; parsed to datetime64 (default: auto-detect)",
    )
    ld.add_argument(
        "--fields", nargs="+", metavar="COL",
        help="columns to inspect and plot (default: all numeric)",
    )
    ld.add_argument("--plot", action="store_true", help="plot selected fields")

    return parser.parse_args()


def _numeric_fields(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Return numeric column names, excluding *exclude*."""
    return [c for c in df.select_dtypes("number").columns if c not in exclude]


def main() -> int:
    args = parse_args()

    if args.mode == "generate":
        csv_path = Path(args.output)
        generate_sample_csv(csv_path, n_rows=args.rows)
        timestamp_col = "timestamp"
        fields_to_plot = ["Icoil", "Ucoil", "teb"]
        df = load_and_inspect(csv_path, timestamp_col, fields=None)
        if args.plot:
            plot_data(df, timestamp_col, fields_to_plot)

    else:  # load
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: {csv_path} not found", file=sys.stderr)
            return 1

        # Auto-detect timestamp column when not given
        from python_magnetrun.magnetdata_pandas import PandasMagnetData
        _probe = PandasMagnetData.fromcsv(str(csv_path))
        all_keys = _probe.getKeys()

        timestamp_col = args.timestamp_col
        if timestamp_col is None:
            for candidate in ("timestamp", "t", "time", "date", "datetime"):
                if candidate in all_keys:
                    timestamp_col = candidate
                    print(f"Auto-detected timestamp column: '{timestamp_col}'")
                    break

        df = load_and_inspect(csv_path, timestamp_col, fields=args.fields)

        if args.plot:
            exclude = [timestamp_col] if timestamp_col else []
            fields = args.fields or _numeric_fields(df, exclude=exclude)
            if not fields:
                print("No numeric columns found to plot.")
            else:
                plot_data(df, timestamp_col, fields)

    return 0


if __name__ == "__main__":
    sys.exit(main())
