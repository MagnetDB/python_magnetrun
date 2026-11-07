#!/usr/bin/env python3
"""Plot raw bdd.csv data against a pupitre aggregate csv.

Pairs compared (bdd <-> aggregate):
teb <-> teb, tsb <-> tsb, ted <-> Tout, tsd <-> (TinH + TinB) / 2.

bdd.csv timestamps are shifted back by ``--shift`` seconds (default 7196 s,
the offset found by compare_bdd_aggregate.py) to match the aggregate clock.

Example
-------
    python plot_bdd_aggregate.py Data/bdd.csv agg.csv --start 2026-09-17
    python plot_bdd_aggregate.py Data/bdd.csv agg.csv --start 2026-09-17 --end 2026-09-19 --save out.png
"""

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAIRS = [
    ("teb", "teb", "teb"),
    ("tsb", "tsb", "tsb"),
    ("ted", "Tout", "Tout"),
    ("tsd", "(TinH + TinB) / 2", "(TinH+TinB)/2"),
]


def load_raw(
    bdd_csv: Path, agg_csv: Path, start: pd.Timestamp, end: pd.Timestamp, shift: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw rows of both files within a time window.

    Parameters
    ----------
    bdd_csv : Path
        bdd.csv export with ``timestamp, teb, tsb, ted, tsd`` columns [°C].
    agg_csv : Path
        Aggregate csv with ``timestamp, teb, tsb, Tout, TinH, TinB`` columns [°C].
    start, end : pandas.Timestamp
        Window on the aggregate time axis.
    shift : int
        Offset [s] subtracted from bdd timestamps.

    Returns
    -------
    bdd : :class:`~pandas.DataFrame`
        Columns ``ts`` (shifted) and ``c0..c3`` following :data:`PAIRS`.
    agg : :class:`~pandas.DataFrame`
        Columns ``ts`` and ``c0..c3`` following :data:`PAIRS`.
    """
    con = duckdb.connect()
    window = f"between '{start}' and '{end}'"
    bdd_cols = ", ".join(f"{b} as c{i}" for i, (b, _, _) in enumerate(PAIRS))
    agg_cols = ", ".join(f"{a} as c{i}" for i, (_, a, _) in enumerate(PAIRS))
    bdd = con.sql(
        f"select * from (select timestamp::TIMESTAMP - to_seconds({shift}) ts, {bdd_cols} "
        f"from read_csv('{bdd_csv}')) where ts {window} order by ts"
    ).df()
    agg = con.sql(
        f"select * from (select timestamp::TIMESTAMP ts, {agg_cols} "
        f"from read_csv('{agg_csv}')) where ts {window} order by ts"
    ).df()
    return bdd, agg


def _break_gaps(ts: pd.Series, values: pd.Series, gap: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN between samples more than ``gap`` seconds apart so lines break."""
    idx = np.flatnonzero(ts.diff().dt.total_seconds().to_numpy()[1:] > gap) + 1
    t = np.insert(ts.to_numpy(), idx, ts.to_numpy()[idx - 1] if len(idx) else [])
    v = np.insert(values.to_numpy(dtype=float), idx, np.nan)
    return t, v


def plot_pairs(bdd: pd.DataFrame, agg: pd.DataFrame, shift: int, save: Path | None = None) -> None:
    """Plot the four bdd/aggregate pairs in stacked panels sharing the time axis.

    Parameters
    ----------
    bdd, agg : :class:`~pandas.DataFrame`
        Output of :func:`load_raw`.
    shift : int
        Applied offset [s], shown in the legend.
    save : Path, optional
        PNG file to write; the interactive window is shown if omitted.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(len(PAIRS), 1, sharex=True, figsize=(14, 10))
    for i, (ax, (b, _, a)) in enumerate(zip(axes, PAIRS, strict=True)):
        ax.plot(*_break_gaps(agg.ts, agg[f"c{i}"]), lw=0.8, label=f"aggregate {a}")
        ax.plot(*_break_gaps(bdd.ts, bdd[f"c{i}"]), lw=0.8, label=f"bdd {b} (shifted {-shift} s)")
        ax.set_ylabel("[°C]")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (aggregate clock)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120, bbox_inches="tight")
        print(f"saved {save}")
    else:
        plt.show()


def main() -> None:
    """Parse arguments, load the window and plot."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bdd_csv", type=Path, help="bdd.csv export")
    parser.add_argument("aggregate_csv", type=Path, help="aggregate csv")
    parser.add_argument("--start", required=True, help="window start (aggregate clock)")
    parser.add_argument("--end", help="window end (default: start + 1 day)")
    parser.add_argument("--shift", type=int, default=7196, help="bdd clock offset [s] (default: 7196)")
    parser.add_argument("--save", type=Path, help="write a PNG instead of opening a window")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else start + pd.Timedelta(1, "D")
    bdd, agg = load_raw(args.bdd_csv, args.aggregate_csv, start, end, args.shift)
    print(f"{start} -> {end}: bdd {len(bdd)} rows, aggregate {len(agg)} rows")
    plot_pairs(bdd, agg, args.shift, args.save)


if __name__ == "__main__":
    main()
