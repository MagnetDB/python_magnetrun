#!/usr/bin/env python3
"""Benchmark the cost of ``utils.narwhals_compat.to_pandas``.

Spike validating the core assumption behind Option B ("boundary
conversion") in `prompts/narwhals-downstream-consumers.plan.md`: that
converting a narwhals/Polars result back to pandas at the point of use is
cheap enough not to matter, because it happens once per CLI/analysis call
(not a hot loop) and rides on top of `PolarsMagnetData`'s already-proven
load-time win (see `prompts/tdms-pupitre-polars-findings.md`).

Measures, on real pupitre fixtures:

- ``to_pandas()`` on a narwhals-wrapped frame from ``getData()``.
- ``to_pandas()`` on a *raw* Polars frame from ``extractData()`` /
  ``extractTimeData()`` (these don't go through the narwhals boundary —
  see the module docstring in ``narwhals_compat.py``).
- The full ``PolarsMagnetData`` vs ``PandasMagnetData`` load
  (``fromtxt()`` + ``addTime()`` + ``Units()`` + ``cleanupData()``), so the
  conversion cost can be read as a fraction of the load-time win it sits
  on top of, not just an absolute number.

Usage::

    python benchmark_to_pandas.py
    python benchmark_to_pandas.py path/to/one.txt --repeat 20
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from python_magnetrun.housing_config import get_housing_config
from python_magnetrun.magnetdata_pandas import PandasMagnetData
from python_magnetrun.magnetdata_polars import PolarsMagnetData
from python_magnetrun.utils.narwhals_compat import to_pandas

DEFAULT_FIXTURES = [
    ("../data/M10_2020.10.23---20_10_41.txt", "M10"),
    ("../data/M9_2019.02.14---23_00_38.txt", "M9"),
]

COLUMN_SIZES = [1, 5, None]  # None = all columns


@dataclass
class _ConvertResult:
    """Timing result for one ``to_pandas()`` call shape.

    Attributes
    ----------
    filepath : str
        Path to the pupitre ``.txt`` fixture.
    source : str
        ``"getData (narwhals)"`` or ``"extractData (raw polars)"``.
    n_columns : int
        Number of columns converted.
    time_min_s : float
        Minimum wall-clock conversion time across repetitions [s].
    time_mean_s : float
        Mean wall-clock conversion time across repetitions [s].
    """

    filepath: str
    source: str
    n_columns: int
    time_min_s: float = float("nan")
    time_mean_s: float = float("nan")


@dataclass
class _LoadResult:
    """Timing result for one full ETL load.

    Attributes
    ----------
    filepath : str
        Path to the pupitre ``.txt`` fixture.
    backend : str
        ``"pandas"`` or ``"polars"``.
    time_s : float
        Wall-clock time for ``fromtxt()`` + ``addTime()`` + ``Units()`` +
        ``cleanupData()`` [s].
    """

    filepath: str
    backend: str
    time_s: float


def _time_calls(fn, repeat: int) -> tuple[float, float]:
    """Run *fn* *repeat* times and return ``(time_min_s, time_mean_s)``."""
    times: list[float] = []
    for _ in range(max(repeat, 1)):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def _prepared(cls, filepath: str, housing: str):
    """Load *filepath* via *cls* and run the real ETL chain used by MagnetRun.fromtxt."""
    cfg = get_housing_config(housing)
    data = cls.fromtxt(filepath, defs_file="pupitre-defs.json")
    data.addTime()
    data.Units()
    available = data.getKeys()
    keys_to_add = {**cfg.pupitre_formula_map, **cfg.get_pupitre_voltage_formulas(available)}
    keys_to_rename = cfg.get_pupitre_rename_map()
    data.cleanupData(keys_to_rename=keys_to_rename, keys_to_add=keys_to_add)
    return data


def _benchmark_conversions(filepath: str, housing: str, repeat: int) -> list[_ConvertResult]:
    """Benchmark to_pandas() on getData()/extractData() results, at each column count."""
    data = _prepared(PolarsMagnetData, filepath, housing)
    all_keys = data.getKeys()
    results = []

    for n in COLUMN_SIZES:
        keys = all_keys if n is None else all_keys[:n]
        n_columns = len(keys)

        t_min, t_mean = _time_calls(lambda keys=keys: to_pandas(data.getData(keys)), repeat)
        results.append(
            _ConvertResult(
                filepath=filepath,
                source="getData (narwhals)",
                n_columns=n_columns,
                time_min_s=t_min,
                time_mean_s=t_mean,
            )
        )

        t_min, t_mean = _time_calls(
            lambda keys=keys: to_pandas(data.extractData(keys)), repeat
        )
        results.append(
            _ConvertResult(
                filepath=filepath,
                source="extractData (raw polars)",
                n_columns=n_columns,
                time_min_s=t_min,
                time_mean_s=t_mean,
            )
        )

    return results


def _benchmark_loads(filepath: str, housing: str, repeat: int) -> list[_LoadResult]:
    """Benchmark the full ETL load for both backends."""
    results = []
    for cls, backend in ((PandasMagnetData, "pandas"), (PolarsMagnetData, "polars")):
        t_min, _ = _time_calls(lambda cls=cls: _prepared(cls, filepath, housing), repeat)
        results.append(_LoadResult(filepath=filepath, backend=backend, time_s=t_min))
    return results


def _print_tables(
    load_results: list[_LoadResult], convert_results: list[_ConvertResult]
) -> None:
    """Print load-time and conversion-time comparison tables."""
    print("\n=== Full ETL load (min time) ===")
    header = f"{'file':<40} {'backend':<8} {'time_s':>10}"
    print(header)
    print("-" * len(header))
    load_by_file: dict[str, float] = {}
    for r in load_results:
        fname = Path(r.filepath).name
        print(f"{fname:<40} {r.backend:<8} {r.time_s:>10.4f}")
        if r.backend == "polars":
            load_by_file[r.filepath] = r.time_s

    print("\n=== to_pandas() conversion cost ===")
    header = (
        f"{'file':<40} {'source':<26} {'cols':>5} "
        f"{'time_min_s':>11} {'time_mean_s':>12} {'% of polars load':>17}"
    )
    print(header)
    print("-" * len(header))
    for r in convert_results:
        fname = Path(r.filepath).name
        load_s = load_by_file.get(r.filepath)
        pct = f"{100 * r.time_min_s / load_s:.3f}%" if load_s else "n/a"
        print(
            f"{fname:<40} {r.source:<26} {r.n_columns:>5} "
            f"{r.time_min_s:>11.6f} {r.time_mean_s:>12.6f} {pct:>17}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark over the given (or default) pupitre files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", help="Pupitre .txt files to benchmark (paired with --housing)"
    )
    parser.add_argument("--housing", default="M9", help="Housing for --files (default: M9)")
    parser.add_argument(
        "--repeat", type=int, default=10, help="Timing repetitions per measurement"
    )
    args = parser.parse_args(argv)

    fixtures = (
        [(f, args.housing) for f in args.files] if args.files else DEFAULT_FIXTURES
    )

    all_loads: list[_LoadResult] = []
    all_converts: list[_ConvertResult] = []
    for filepath, housing in fixtures:
        path = Path(filepath)
        if not path.is_file():
            print(f"skipping {filepath}: file not found")
            continue
        print(f"\n=== {path.name} (housing={housing}) ===")
        all_loads.extend(_benchmark_loads(str(path), housing, args.repeat))
        all_converts.extend(_benchmark_conversions(str(path), housing, args.repeat))

    _print_tables(all_loads, all_converts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
