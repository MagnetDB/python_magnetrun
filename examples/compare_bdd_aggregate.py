#!/usr/bin/env python3
"""Compare a temperature column between a bdd.csv export and a pupitre aggregate csv.

The two sources are expected to carry the same signal (e.g. ``teb``) with a
possible clock offset (bdd.csv in local time vs aggregate in UTC) and a
different resolution (aggregate values truncated to 0.1 °C).

Example
-------
    python compare_bdd_aggregate.py Data/bdd.csv 2026-aggregate-teb-tsb-Tout-TinH-TinB.csv
    python compare_bdd_aggregate.py Data/bdd.csv agg.csv --column tsb --plot --save out/
"""

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt


def load_tables(con: duckdb.DuckDBPyConnection, bdd_csv: Path, agg_csv: Path, column: str) -> None:
    """Load both csv files into tables ``b`` (bdd) and ``a`` (aggregate).

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        In-memory DuckDB connection.
    bdd_csv : Path
        bdd.csv export (``timestamp`` column plus sensor columns).
    agg_csv : Path
        Aggregate csv (``timestamp`` column plus sensor columns).
    column : str
        Name of the column to compare, e.g. ``"teb"`` [°C].

    Returns
    -------
    None
        Tables ``b`` and ``a`` with columns ``ts`` and ``v`` are created;
        duplicated timestamps in bdd.csv are averaged.
    """
    con.sql(
        f'create or replace table b as select timestamp::TIMESTAMP ts, avg("{column}") v '
        f"from read_csv('{bdd_csv}') group by 1"
    )
    con.sql(
        f'create or replace table a as select timestamp::TIMESTAMP ts, "{column}" v '
        f"from read_csv('{agg_csv}')"
    )
    for name, src in (("b1m", "b"), ("a1m", "a")):
        con.sql(
            f"create or replace table {name} as select time_bucket(interval 1 minute, ts) ts, "
            f"avg(v) v from {src} group by 1"
        )


def summarize(con: duckdb.DuckDBPyConnection, bdd_csv: Path) -> None:
    """Print row count, time span and min/max/mean of each source.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection holding tables ``a`` and ``b``.
    bdd_csv : Path
        bdd.csv export, re-read to count duplicated timestamps.

    Returns
    -------
    None
    """
    dup = con.sql(
        f"select count(*) - count(distinct timestamp) from read_csv('{bdd_csv}')"
    ).fetchone()[0]
    print(f"{'':10} {'rows':>9} {'start':>20} {'end':>20} {'min':>7} {'max':>7} {'mean':>7}")
    for label, t in (("bdd", "b"), ("aggregate", "a")):
        n, t0, t1, vmin, vmax, vmean = con.sql(
            f"select count(*), min(ts), max(ts), min(v), max(v), avg(v) from {t}"
        ).fetchone()
        print(f"{label:10} {n:9d} {t0!s:>20} {t1!s:>20} {vmin:7.2f} {vmax:7.2f} {vmean:7.2f}")
    print(f"bdd duplicated timestamps (averaged): {dup}")


def _corr_at(con: duckdb.DuckDBPyConnection, a: str, b: str, seconds: int) -> tuple:
    """Return (n, corr, bias, std) of ``b - a`` with ``b.ts = a.ts + seconds``."""
    return con.sql(
        f"select count(*), corr(a.v, b.v), avg(b.v - a.v), stddev(b.v - a.v) "
        f"from {a} a join {b} b on b.ts = a.ts + to_seconds({seconds})"
    ).fetchone()


def scan_lag(con: duckdb.DuckDBPyConnection, lag_range: tuple[int, int, int]) -> tuple[int, list[tuple]]:
    """Find the time offset that best aligns bdd on the aggregate.

    A coarse scan on 1-minute means is refined to 1-minute steps around the
    best coarse lag, then to 1-second steps on the raw data.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection holding tables ``a``, ``b``, ``a1m`` and ``b1m``.
    lag_range : tuple of int
        ``(start, stop, step)`` of the coarse scan [min], ``stop`` included.

    Returns
    -------
    best : int
        Best lag [s]; bdd timestamp = aggregate timestamp + ``best``.
    coarse : list of tuple
        ``(lag [min], n, corr, bias, std)`` for each coarse lag.
    """
    start, stop, step = lag_range
    print("\nlag scan on 1-minute means (bdd.ts = aggregate.ts + lag), lag in minutes")
    print(f"{'lag':>5} {'n':>7} {'corr':>7} {'bias':>7} {'std':>7}")
    coarse = []
    for m in range(start, stop + 1, step):
        n, r, bias, std = _corr_at(con, "a1m", "b1m", 60 * m)
        coarse.append((m, n, r, bias, std))
        print(f"{m:5d} {n:7d} {r:7.3f} {bias:7.3f} {std:7.3f}")
    best_m = max(coarse, key=lambda c: c[2])[0]
    best_m = max(
        range(best_m - step, best_m + step + 1), key=lambda m: _corr_at(con, "a1m", "b1m", 60 * m)[1]
    )

    print(f"\n1-second scan around {best_m} min")
    print(f"{'lag[s]':>7} {'n':>8} {'corr':>8} {'bias':>7} {'std':>7}")
    fine = []
    for s in range(60 * best_m - 60, 60 * best_m + 61, 2):
        n, r, bias, std = _corr_at(con, "a", "b", s)
        fine.append((s, r))
        if s % 10 == 0:
            print(f"{s:7d} {n:8d} {r:8.5f} {bias:7.4f} {std:7.4f}")
    best = max(fine, key=lambda f: f[1])[0]
    best = max(range(best - 1, best + 2), key=lambda s: _corr_at(con, "a", "b", s)[1])
    print(f"best lag: {best} s ({best / 3600:.3f} h)")
    return best, coarse


def compare_at_lag(con: duckdb.DuckDBPyConnection, lag: int) -> None:
    """Print agreement statistics once bdd is shifted by ``lag``.

    Creates table ``m`` (``ts``, ``ta``, ``tb``, ``d = tb - ta``) on the
    aggregate time axis.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection holding tables ``a`` and ``b``.
    lag : int
        Offset [s]; bdd timestamp = aggregate timestamp + ``lag``.

    Returns
    -------
    None
    """
    con.sql(
        "create or replace table m as select a.ts, a.v ta, b.v tb, b.v - a.v d "
        f"from a join b on b.ts = a.ts + to_seconds({lag})"
    )
    n, r, bias, std, p999, dmax, rnd, flo = con.sql(
        "select count(*), corr(ta, tb), avg(d), stddev(d), quantile_cont(abs(d), 0.999), "
        "max(abs(d)), avg((round(tb, 1) = ta)::INT), avg((floor(tb * 10) / 10 = ta)::INT) from m"
    ).fetchone()
    print(f"\nat lag {lag} s: d = bdd - aggregate")
    print(f"  matched pairs   : {n}")
    print(f"  corr            : {r:.5f}")
    print(f"  bias / std      : {bias:.4f} / {std:.4f}")
    print(f"  p99.9 |d| / max : {p999:.3f} / {dmax:.3f}")
    print(f"  aggregate == round(bdd, 1): {rnd:.1%}   == floor(bdd, 1): {flo:.1%}")

    print("\nper month (aggregate time)")
    print(f"{'month':>7} {'n':>8} {'bias':>7} {'std':>7} {'corr':>7}")
    for mo, n, bias, std, r in con.sql(
        "select strftime(ts, '%Y-%m'), count(*), avg(d), stddev(d), corr(ta, tb) from m group by 1 order by 1"
    ).fetchall():
        print(f"{mo:>7} {n:8d} {bias:7.3f} {std:7.3f} {r:7.4f}")

    print("\nworst differences")
    print(con.sql('select ts, ta as "aggregate", tb as bdd, d from m order by abs(d) desc limit 8'))

    unmatched = con.sql(
        f"select count(*) from a where ts + to_seconds({lag}) between (select min(ts) from b) and "
        f"(select max(ts) from b) and not exists (select 1 from b where b.ts = a.ts + to_seconds({lag}))"
    ).fetchone()[0]
    print(f"aggregate rows within the bdd span without a bdd match: {unmatched}")


def plot_comparison(
    con: duckdb.DuckDBPyConnection,
    lag: int,
    coarse: list[tuple],
    column: str,
    start: str | None = None,
    end: str | None = None,
    save: Path | None = None,
) -> None:
    """Plot overlay, difference, difference histogram and lag scan.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection holding table ``m`` from :func:`compare_at_lag`.
    lag : int
        Applied offset [s], shown in titles.
    coarse : list of tuple
        Coarse lag scan from :func:`scan_lag`.
    column : str
        Compared column name, used for labels.
    start, end : str, optional
        Time window on the aggregate axis, e.g. ``"2026-07-01"``.
    save : Path, optional
        Directory where PNG files are written; figures are shown if omitted.

    Returns
    -------
    None
    """
    where = []
    if start:
        where.append(f"ts >= '{start}'")
    if end:
        where.append(f"ts < '{end}'")
    cond = f"where {' and '.join(where)}" if where else ""
    df = con.sql(
        f"select time_bucket(interval 1 minute, ts) ts, avg(ta) ta, avg(tb) tb, avg(d) d "
        f"from m {cond} group by 1 order by 1"
    ).df()
    d = con.sql(f"select d from m {cond}").df()["d"]

    figs = {}
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.ts, df.ta, ".", ms=1, label="aggregate")
    ax.plot(df.ts, df.tb, ".", ms=1, label=f"bdd shifted by {-lag} s")
    ax.set(xlabel="time (aggregate)", ylabel=f"{column} [°C]", title=f"{column}: 1-minute means")
    ax.legend(markerscale=10)
    figs["overlay"] = fig

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.ts, df.d, ".", ms=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="time (aggregate)", ylabel="bdd - aggregate [°C]", title=f"{column}: difference (1-minute means)")
    figs["difference"] = fig

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d, bins=200, range=(-0.5, 0.5), log=True)
    ax.set(xlabel="bdd - aggregate [°C]", ylabel="count", title=f"{column}: difference at lag {lag} s")
    figs["histogram"] = fig

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([c[0] for c in coarse], [c[2] for c in coarse], "o-")
    ax.set(xlabel="lag [min]", ylabel="correlation", title=f"{column}: lag scan (1-minute means)")
    ax.grid(True)
    figs["lag_scan"] = fig

    if save:
        save.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            path = save / f"{column}_{name}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"saved {path}")
        plt.close("all")
    else:
        plt.show()


def main() -> None:
    """Parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bdd_csv", type=Path, help="bdd.csv export")
    parser.add_argument("aggregate_csv", type=Path, help="aggregate csv")
    parser.add_argument("--column", default="teb", help="column to compare (default: teb)")
    parser.add_argument(
        "--lag-range",
        type=int,
        nargs=3,
        default=(-180, 180, 10),
        metavar=("START", "STOP", "STEP"),
        help="coarse lag scan [min] (default: -180 180 10)",
    )
    parser.add_argument("--plot", action="store_true", help="plot the comparison")
    parser.add_argument("--start", help="plot window start (aggregate time)")
    parser.add_argument("--end", help="plot window end (aggregate time)")
    parser.add_argument("--save", type=Path, help="write PNG files to this directory instead of showing")
    args = parser.parse_args()

    con = duckdb.connect()
    load_tables(con, args.bdd_csv, args.aggregate_csv, args.column)
    summarize(con, args.bdd_csv)
    lag, coarse = scan_lag(con, tuple(args.lag_range))
    compare_at_lag(con, lag)
    if args.plot:
        plot_comparison(con, lag, coarse, args.column, args.start, args.end, args.save)


if __name__ == "__main__":
    main()
