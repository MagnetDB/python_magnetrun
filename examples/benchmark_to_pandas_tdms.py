#!/usr/bin/env python3
"""Benchmark the cost of ``utils.narwhals_compat.to_pandas`` for TDMS groups.

Companion to ``examples/benchmark_to_pandas.py`` (the pupitre version).
TDMS groups matter more for this question than pupitre tables: our real
fixture has groups from 4 to 32 channels, all at 432 000 rows, much larger
than any pupitre table benchmarked so far — exactly the "large data" case
`ROADMAP.md` item 4.3 was originally worried about.

Measures, per group, on a real TDMS fixture:

- ``to_pandas()`` on a narwhals-wrapped frame from ``TdmsMagnetData.getData()``.
- ``to_pandas()`` on the *raw* Polars group DataFrame (the shape
  ``extractData()``/``saveData()``/``stats()``/``plotData()`` each convert
  internally — see ``magnetdata_tdms.py``).

Also times ``TdmsGroup.as_dataframe()`` itself (the group *load*, same
measurement as ``examples/benchmark_tdms_polars.py``), so the conversion
cost can be read as a fraction of load time, not just an absolute number.

Usage::

    python benchmark_to_pandas_tdms.py
    python benchmark_to_pandas_tdms.py path/to/one.tdms --defs-file pigbrother-defs.json --repeat 10
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from nptdms import TdmsFile

from python_magnetrun.magnetdata import _fromtdms
from python_magnetrun.utils.narwhals_compat import to_pandas

DEFAULT_FIXTURE = "../data/M9_Default_200921-123303_Courants50Hz.tdms"
DEFAULT_DEFS_FILE = "pigbrother-defs.json"


@dataclass
class _LoadResult:
    """Timing result for one group's ``as_dataframe()`` load.

    Attributes
    ----------
    filepath : str
        Path to the ``.tdms`` fixture.
    group : str
        TDMS group name.
    n_channels : int
        Number of channels in the group [dimensionless].
    n_rows : int
        Row count of the group.
    time_min_s : float
        Minimum wall-clock load time across repetitions [s].
    """

    filepath: str
    group: str
    n_channels: int
    n_rows: int
    time_min_s: float


@dataclass
class _ConvertResult:
    """Timing result for one ``to_pandas()`` call shape.

    Attributes
    ----------
    filepath : str
        Path to the ``.tdms`` fixture.
    group : str
        TDMS group name.
    source : str
        ``"getData (narwhals)"`` or ``"raw group DataFrame"``.
    n_channels : int
        Number of channels converted [dimensionless].
    n_rows : int
        Row count of the group.
    time_min_s : float
        Minimum wall-clock conversion time across repetitions [s].
    time_mean_s : float
        Mean wall-clock conversion time across repetitions [s].
    """

    filepath: str
    group: str
    source: str
    n_channels: int
    n_rows: int
    time_min_s: float
    time_mean_s: float


def _time_calls(fn, repeat: int) -> tuple[float, float]:
    """Run *fn* *repeat* times and return ``(time_min_s, time_mean_s)``."""
    times: list[float] = []
    for _ in range(max(repeat, 1)):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def _benchmark_loads(filepath: str, repeat: int) -> list[_LoadResult]:
    """Time ``TdmsGroup.as_dataframe()`` directly, per group.

    Mirrors the measurement in ``examples/benchmark_tdms_polars.py`` —
    included here only for context, to size the conversion cost below
    against it.
    """
    results = []
    with TdmsFile.open(filepath) as tdms_file:
        for group in tdms_file.groups():
            # Match the group-name rewrite in magnetdata._fromtdms() so
            # results line up with _benchmark_conversions()'s group keys.
            gname = group.name.replace(" ", "_").replace("_et_Ref.", "")
            if gname == "Infos":
                continue
            channels = list(group.channels())
            n_channels = len(channels)
            n_rows = len(channels[0]) if channels else 0
            t_min, _ = _time_calls(
                lambda group=group: group.as_dataframe(
                    time_index=False, absolute_time=False, scaled_data=True
                ),
                repeat,
            )
            results.append(
                _LoadResult(
                    filepath=filepath,
                    group=gname,
                    n_channels=n_channels,
                    n_rows=n_rows,
                    time_min_s=t_min,
                )
            )
    return results


def _benchmark_conversions(
    filepath: str, defs_file: str, repeat: int
) -> list[_ConvertResult]:
    """Time ``to_pandas()`` on ``getData()`` (narwhals) and the raw group DataFrame."""
    data = _fromtdms(filepath, defs_file=defs_file)
    results = []

    for gname in list(data.Groups):
        if gname == "Infos":
            continue
        data.getTdmsData(gname, None)  # force-load once; cached by _ensure_group_loaded
        raw_df = data.Data[gname]
        n_channels = raw_df.width
        n_rows = raw_df.height

        t_min, t_mean = _time_calls(
            lambda gname=gname: to_pandas(data.getData(gname)), repeat
        )
        results.append(
            _ConvertResult(
                filepath=filepath,
                group=gname,
                source="getData (narwhals)",
                n_channels=n_channels,
                n_rows=n_rows,
                time_min_s=t_min,
                time_mean_s=t_mean,
            )
        )

        t_min, t_mean = _time_calls(lambda raw_df=raw_df: to_pandas(raw_df), repeat)
        results.append(
            _ConvertResult(
                filepath=filepath,
                group=gname,
                source="raw group DataFrame",
                n_channels=n_channels,
                n_rows=n_rows,
                time_min_s=t_min,
                time_mean_s=t_mean,
            )
        )

    return results


def _print_tables(
    load_results: list[_LoadResult], convert_results: list[_ConvertResult]
) -> None:
    """Print load-time and conversion-time comparison tables."""
    print("\n=== Group load: TdmsGroup.as_dataframe() (min time) ===")
    header = f"{'group':<32} {'ch':>4} {'rows':>8} {'time_s':>10}"
    print(header)
    print("-" * len(header))
    load_by_group: dict[str, float] = {}
    for r in load_results:
        print(f"{r.group:<32} {r.n_channels:>4} {r.n_rows:>8} {r.time_min_s:>10.4f}")
        load_by_group[r.group] = r.time_min_s

    print("\n=== to_pandas() conversion cost ===")
    header = (
        f"{'group':<32} {'source':<22} {'ch':>4} {'rows':>8} "
        f"{'time_min_s':>11} {'time_mean_s':>12} {'% of load':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in convert_results:
        load_s = load_by_group.get(r.group)
        pct = f"{100 * r.time_min_s / load_s:.3f}%" if load_s else "n/a"
        print(
            f"{r.group:<32} {r.source:<22} {r.n_channels:>4} {r.n_rows:>8} "
            f"{r.time_min_s:>11.6f} {r.time_mean_s:>12.6f} {pct:>10}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark over the given (or default) TDMS file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file", nargs="?", default=DEFAULT_FIXTURE, help="TDMS file to benchmark"
    )
    parser.add_argument(
        "--defs-file", default=DEFAULT_DEFS_FILE, help="Field-definition JSON file"
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="Timing repetitions per measurement"
    )
    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.is_file():
        print(f"file not found: {args.file}")
        return 1

    print(f"\n=== {path.name} ({path.stat().st_size / 1024**2:.1f} MiB) ===")
    load_results = _benchmark_loads(str(path), args.repeat)
    convert_results = _benchmark_conversions(str(path), args.defs_file, args.repeat)
    _print_tables(load_results, convert_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
