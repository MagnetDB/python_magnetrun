#!/usr/bin/env python3
"""Benchmark TDMS group export: pandas vs polars (Trophime/npTDMS fork).

Spike for `ROADMAP.md` item 4.3 / `prompts/mrun-cache-implementation.plan.md`
Phase "1+2b-tdms". Mirrors the exact access pattern used in production by
:meth:`~python_magnetrun.magnetdata_tdms.TdmsMagnetData._ensure_group_loaded`
(``TdmsFile.open(name)`` then ``group.as_dataframe(time_index=False,
absolute_time=False, scaled_data=True)``), comparing it against the pandas
export path called directly (``nptdms.export.pandas_export.from_group``).

Requires the polars-enabled fork, not the PyPI ``nptdms`` release::

    pip install "git+https://github.com/Trophime/npTDMS.git@master" polars pandas

Usage::

    python benchmark_tdms_polars.py
    python benchmark_tdms_polars.py path/to/one.tdms path/to/two.tdms --repeat 5
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from nptdms import TdmsFile
    from nptdms.export import pandas_export, polars_export
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        "This script requires the Trophime/npTDMS fork with polars support:\n"
        '  pip install "git+https://github.com/Trophime/npTDMS.git@master" polars pandas\n'
        f"Import error: {exc}"
    )

DEFAULT_FIXTURES = [
    "data/M9_Default_200921-123303_Courants50Hz.tdms",
    "../apps/dashboards/magnetdb/tests/M9_Archive_251202-1430.tdms",
]


@dataclass
class _GroupResult:
    """Timing/memory result for one TDMS group, one export backend.

    Attributes
    ----------
    filepath : str
        Path to the ``.tdms`` file.
    group : str
        TDMS group name.
    backend : str
        ``"pandas"`` or ``"polars"``.
    n_channels : int
        Number of channels in the group [dimensionless].
    n_rows : int
        Row count of the exported DataFrame.
    time_min_s : float
        Minimum wall-clock export time across repetitions [s].
    time_mean_s : float
        Mean wall-clock export time across repetitions [s].
    peak_memory_mib : float
        Peak traced memory during one export call [MiB].
    error : str
        Non-empty when the export failed.
    """

    filepath: str
    group: str
    backend: str
    n_channels: int = 0
    n_rows: int = 0
    time_min_s: float = float("nan")
    time_mean_s: float = float("nan")
    peak_memory_mib: float = float("nan")
    error: str = ""


def _time_export(fn, repeat: int) -> tuple[float, float, object]:
    """Run *fn* *repeat* times and return ``(time_min_s, time_mean_s, last_result)``."""
    times: list[float] = []
    result = None
    for _ in range(max(repeat, 1)):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times), result


def _peak_memory_mib(fn) -> float:
    """Return peak traced memory [MiB] of one call to *fn*."""
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024**2


def _benchmark_group(filepath: str, group_name: str, group, repeat: int) -> list[_GroupResult]:
    """Benchmark pandas vs polars export for one TDMS group.

    Parameters
    ----------
    filepath : str
        Path to the ``.tdms`` file (for the result record only).
    group_name : str
        Normalised group name (for the result record only).
    group : nptdms.tdms.TdmsGroup
        Group object to export.
    repeat : int
        Timing repetitions per backend.

    Returns
    -------
    list of _GroupResult
        One record per backend (``"pandas"``, ``"polars"``).
    """
    n_channels = len(list(group.channels()))
    results = []

    backends = (
        (
            "pandas",
            lambda: pandas_export.from_group(
                group, time_index=False, absolute_time=False, scaled_data=True
            ),
        ),
        (
            "polars",
            lambda: group.as_dataframe(
                time_index=False, absolute_time=False, scaled_data=True
            ),
        ),
    )

    for backend, fn in backends:
        result = _GroupResult(
            filepath=filepath, group=group_name, backend=backend, n_channels=n_channels
        )
        try:
            t_min, t_mean, df = _time_export(fn, repeat)
            result.time_min_s = t_min
            result.time_mean_s = t_mean
            result.peak_memory_mib = _peak_memory_mib(fn)
            result.n_rows = len(df) if backend == "pandas" else df.height
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
        results.append(result)

    return results


def _sanity_check(group, channel_name: str) -> None:
    """Compare pandas vs polars export for one channel via :func:`numpy.allclose`.

    Parameters
    ----------
    group : nptdms.tdms.TdmsGroup
        Group containing *channel_name*.
    channel_name : str
        Channel to compare across both export backends.
    """
    channel = group[channel_name]
    try:
        pandas_df = pandas_export.from_channel(
            channel, time_index=False, absolute_time=False, scaled_data=True
        )
        polars_df = polars_export.from_channel(
            channel, time_index=False, absolute_time=False, scaled_data=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  sanity check skipped for {channel_name!r}: {exc}")
        return

    pandas_arr = pandas_df.iloc[:, 0].to_numpy()
    polars_arr = polars_df.to_series(0).to_numpy()
    if not np.allclose(pandas_arr, polars_arr, equal_nan=True):
        raise AssertionError(f"pandas/polars mismatch for channel {channel_name!r}")
    print(f"  sanity check OK: channel {channel_name!r} ({len(pandas_arr)} samples)")


def _print_table(results: list[_GroupResult]) -> None:
    """Print a comparison table to stdout.

    Parameters
    ----------
    results : list of _GroupResult
        Records to print, in order.
    """
    header = (
        f"{'file':<32} {'group':<32} {'backend':<8} {'ch':>4} {'rows':>8} "
        f"{'time_min_s':>11} {'time_mean_s':>12} {'peak_MiB':>9}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        fname = Path(r.filepath).name
        if r.error:
            print(f"{fname:<32} {r.group:<32} {r.backend:<8} ERROR: {r.error}")
            continue
        print(
            f"{fname:<32} {r.group:<32} {r.backend:<8} {r.n_channels:>4} {r.n_rows:>8} "
            f"{r.time_min_s:>11.4f} {r.time_mean_s:>12.4f} {r.peak_memory_mib:>9.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark over the given (or default) TDMS files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", default=DEFAULT_FIXTURES, help="TDMS files to benchmark"
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="Timing repetitions per group/backend"
    )
    args = parser.parse_args(argv)

    all_results: list[_GroupResult] = []
    for filepath in args.files:
        path = Path(filepath)
        if not path.is_file():
            print(f"skipping {filepath}: file not found")
            continue

        print(f"\n=== {path.name} ({path.stat().st_size / 1024**2:.1f} MiB) ===")
        with TdmsFile.open(str(path)) as tdms_file:
            for group in tdms_file.groups():
                if group.name == "Infos":
                    continue
                gname = group.name.replace(" ", "_")
                all_results.extend(_benchmark_group(str(path), gname, group, args.repeat))

                channels = list(group.channels())
                if channels:
                    _sanity_check(group, channels[0].name)

    _print_table(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
