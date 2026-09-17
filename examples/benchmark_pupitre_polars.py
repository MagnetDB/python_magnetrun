#!/usr/bin/env python3
"""Benchmark pupitre ``.txt`` loading: pandas vs polars.

Spike for `ROADMAP.md` item 4.3, extending `benchmark_tdms_polars.py` to the
pupitre loading path. Mirrors the exact call used in production by
:meth:`~python_magnetrun.readers.csv_readers.PupitreReader.read`
(``pd.read_csv(path, sep=r"\\s+", skiprows=1, on_bad_lines="warn")``),
compared against ``pl.read_csv(path, separator="\\t", skip_rows=1)``.

Format quirks confirmed on the real fixtures before writing this:

- pandas' ``sep=r"\\s+"`` is a regex (arbitrary whitespace runs); polars'
  ``separator`` accepts only a single literal character. The real fixtures
  are strictly single-tab-separated data rows (no padded/variable
  whitespace) — a fact confirmed empirically for these files, not a general
  guarantee for every pupitre file.
- Every data line ends with a trailing tab, producing one extra empty field
  beyond the real column count. Pandas' whitespace-regex split silently
  absorbs it; polars' literal-tab split does not.
- One fixture (``M9_2019.02.14---23_00_38.txt``) has a **malformed header
  row**: doubled tabs after ``Date`` and ``Time`` (``Date\\t\\tTime\\t\\t...``)
  that don't match the single-tab data rows below it. Pandas' regex collapses
  the doubled tabs, parsing the header consistently with the data; polars'
  literal split would not, producing misaligned/duplicate-named columns.
- **Header-only files** (metadata line + header line, zero data rows) are
  real and found in production data (two files under 250 bytes in a
  representative sample). Pandas returns a correctly-shaped 0-row DataFrame;
  ``pl.read_csv`` raises ``NoDataError`` when nothing is left to read after
  ``skip_rows``, which this script catches to build an empty DataFrame with
  the parsed header names instead.

To handle these quirks without silently guessing, this script parses the
header line itself with the same ``\\s+`` regex pandas uses, reads the data
body with polars *without* header inference (``has_header=False``), asserts
the resulting column count is exactly one more than the parsed header names
(the trailing-tab artifact — anything else means an unhandled format change),
drops that trailing column after confirming it is all-null, then applies the
correctly-parsed names.

``on_bad_lines="warn"`` has no exact polars equivalent and is not exercised
by these fixtures, so it is intentionally left untested here.

Memory is measured differently per backend, and the two numbers are **not**
directly comparable: pandas/numpy buffers pass through CPython's memory
allocator, so :mod:`tracemalloc` gives a real peak-during-parsing figure.
polars' ``read_csv`` allocates natively in Rust, invisible to
:mod:`tracemalloc` entirely (it would otherwise silently under-report by two
orders of magnitude). The polars figure is instead the resulting
DataFrame's ``estimated_size()`` — the final result's size, not a peak
observed during parsing.

Requires polars alongside the project's normal dependencies::

    pip install polars

Usage::

    python benchmark_pupitre_polars.py
    python benchmark_pupitre_polars.py path/to/one.txt path/to/two.txt --repeat 5
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(f"This script requires polars: pip install polars\nImport error: {exc}")

DEFAULT_FIXTURES = [
    "../data/M10_2020.10.23---20_10_41.txt",
    "../data/M9_2019.02.14---23_00_38.txt",
]

SEP = r"\s+"
SEPARATOR = "\t"
SKIP_ROWS = 1


@dataclass
class _FileResult:
    """Timing/memory result for one pupitre file, one backend.

    Attributes
    ----------
    filepath : str
        Path to the pupitre ``.txt`` file.
    backend : str
        ``"pandas"`` or ``"polars"``.
    n_columns : int
        Number of data columns after dropping the trailing null column
        (polars only; pandas never has it) [dimensionless].
    n_rows : int
        Row count of the parsed DataFrame.
    time_min_s : float
        Minimum wall-clock parse time across repetitions [s].
    time_mean_s : float
        Mean wall-clock parse time across repetitions [s].
    memory_mib : float
        Memory figure [MiB] — meaning depends on *memory_metric*.
    memory_metric : str
        ``"tracemalloc_peak"`` (pandas) or ``"estimated_size"`` (polars) —
        see module docstring; the two are not directly comparable.
    error : str
        Non-empty when parsing failed.
    """

    filepath: str
    backend: str
    n_columns: int = 0
    n_rows: int = 0
    time_min_s: float = float("nan")
    time_mean_s: float = float("nan")
    memory_mib: float = float("nan")
    memory_metric: str = ""
    error: str = ""


def _read_pandas(path: str) -> pd.DataFrame:
    """Parse *path* the way :meth:`PupitreReader.read` does."""
    return pd.read_csv(path, sep=SEP, skiprows=SKIP_ROWS, on_bad_lines="warn")


def _parse_header(path: str) -> list[str]:
    """Parse the column-name row with the same whitespace-regex pandas uses.

    Needed because the raw header row on at least one real fixture has
    doubled tabs that don't match the single-tab data rows below it —
    polars' literal-separator ``read_csv`` cannot parse it correctly, but
    ``pandas``' ``sep=r"\\s+"`` collapses it as intended.
    """
    with open(path, encoding="utf-8") as f:
        for _ in range(SKIP_ROWS):
            f.readline()
        header_line = f.readline()
    return re.split(SEP, header_line.strip())


def _read_polars(path: str) -> pl.DataFrame:
    """Parse *path* with polars, using a pandas-equivalent header parse.

    The data body is read without polars' own header inference (which only
    understands a single literal separator) and the correctly-parsed column
    names are applied afterwards. The trailing-tab artifact — one extra
    all-null column at the end of every data row — is dropped after
    confirming the column-count mismatch is exactly one and the column is
    in fact all-null.

    Header-only files (no data rows past the header) make ``pl.read_csv``
    raise :class:`polars.exceptions.NoDataError`, unlike pandas which returns
    a correctly-shaped 0-row DataFrame; that case is handled explicitly.
    """
    columns = _parse_header(path)
    try:
        df = pl.read_csv(path, separator=SEPARATOR, skip_rows=SKIP_ROWS + 1, has_header=False)
    except pl.exceptions.NoDataError:
        return pl.DataFrame({col: [] for col in columns})

    n_extra = df.width - len(columns)
    if n_extra != 1:
        raise AssertionError(
            f"expected exactly one trailing artifact column, got {n_extra} "
            f"(header has {len(columns)} names, body has {df.width} columns)"
        )
    trailing = df.columns[-1]
    if not df[trailing].is_null().all():
        raise AssertionError(
            f"expected trailing column {trailing!r} to be all-null "
            "(format assumption changed) — refusing to silently drop it"
        )
    df = df.drop(trailing)
    df.columns = columns
    return df


def _time_read(fn, path: str, repeat: int) -> tuple[float, float, object]:
    """Run *fn(path)* *repeat* times and return ``(time_min_s, time_mean_s, last_result)``."""
    times: list[float] = []
    result = None
    for _ in range(max(repeat, 1)):
        gc.collect()
        t0 = time.perf_counter()
        result = fn(path)
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times), result


def _peak_memory_mib(fn, path: str) -> float:
    """Return peak traced memory [MiB] of one call to ``fn(path)`` (pandas only).

    :mod:`tracemalloc` only sees allocations that pass through CPython's
    memory allocator — accurate for pandas/numpy buffers, but blind to
    polars' native Rust allocations (see :func:`_final_size_mib`).
    """
    gc.collect()
    tracemalloc.start()
    try:
        fn(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024**2


def _final_size_mib(path: str) -> float:
    """Return the parsed polars DataFrame's ``estimated_size()`` [MiB].

    Not a peak-during-parsing measurement like the pandas figure —
    :mod:`tracemalloc` cannot see polars' native allocations at all, so this
    reports the final result's size instead.
    """
    df = _read_polars(path)
    return df.estimated_size() / 1024**2


def _benchmark_file(filepath: str, repeat: int) -> list[_FileResult]:
    """Benchmark pandas vs polars loading for one pupitre file."""
    results = []

    for backend, fn in (("pandas", _read_pandas), ("polars", _read_polars)):
        result = _FileResult(filepath=filepath, backend=backend)
        try:
            t_min, t_mean, df = _time_read(fn, filepath, repeat)
            result.time_min_s = t_min
            result.time_mean_s = t_mean
            if backend == "pandas":
                result.memory_mib = _peak_memory_mib(fn, filepath)
                result.memory_metric = "tracemalloc_peak"
            else:
                result.memory_mib = _final_size_mib(filepath)
                result.memory_metric = "estimated_size"
            result.n_rows = len(df) if backend == "pandas" else df.height
            result.n_columns = df.shape[1] if backend == "pandas" else df.width
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
        results.append(result)

    return results


def _sanity_check(filepath: str, n_columns: int = 5) -> None:
    """Compare the first *n_columns* numeric columns via :func:`numpy.allclose`."""
    pandas_df = _read_pandas(filepath)
    polars_df = _read_polars(filepath)

    if list(pandas_df.columns) != polars_df.columns:
        raise AssertionError(
            f"column mismatch after dropping trailing column: "
            f"pandas={list(pandas_df.columns)!r} polars={polars_df.columns!r}"
        )

    checked = 0
    for col in pandas_df.columns:
        if checked >= n_columns:
            break
        if not pd.api.types.is_numeric_dtype(pandas_df[col]):
            continue
        pandas_arr = pandas_df[col].to_numpy()
        polars_arr = polars_df[col].to_numpy()
        if not np.allclose(pandas_arr, polars_arr, equal_nan=True):
            raise AssertionError(f"pandas/polars mismatch for column {col!r}")
        checked += 1

    print(f"  sanity check OK: {checked} numeric columns match ({len(pandas_df)} rows)")


def _print_table(results: list[_FileResult]) -> None:
    """Print a comparison table to stdout."""
    header = (
        f"{'file':<40} {'backend':<8} {'cols':>5} {'rows':>8} "
        f"{'time_min_s':>11} {'time_mean_s':>12} {'mem_MiB':>9} {'mem_metric':>17}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        fname = Path(r.filepath).name
        if r.error:
            print(f"{fname:<40} {r.backend:<8} ERROR: {r.error}")
            continue
        print(
            f"{fname:<40} {r.backend:<8} {r.n_columns:>5} {r.n_rows:>8} "
            f"{r.time_min_s:>11.4f} {r.time_mean_s:>12.4f} {r.memory_mib:>9.2f} "
            f"{r.memory_metric:>17}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark over the given (or default) pupitre files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", default=DEFAULT_FIXTURES, help="Pupitre .txt files to benchmark"
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="Timing repetitions per file/backend"
    )
    args = parser.parse_args(argv)

    all_results: list[_FileResult] = []
    for filepath in args.files:
        path = Path(filepath)
        if not path.is_file():
            print(f"skipping {filepath}: file not found")
            continue

        print(f"\n=== {path.name} ({path.stat().st_size / 1024**2:.2f} MiB) ===")
        try:
            _sanity_check(str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"  sanity check FAILED: {exc}")

        all_results.extend(_benchmark_file(str(path), args.repeat))

    _print_table(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
