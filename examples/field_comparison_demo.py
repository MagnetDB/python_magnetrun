#!/usr/bin/env python3
"""Example: demonstrate field_comparison.py and benchmark the lag algorithms.

Uses synthetic pupitre / pigbrother DataFrames (no real Overview/Archive/
pupitre files are required) to:

1. Discover pupitre <-> pigbrother aliased fields
   (:func:`~python_magnetrun.analysis.field_comparison.discover_pupitre_pigbrother_fields`).
2. Compute a single reference lag for a source and compare two fields
   against it -- the same workflow
   :func:`~python_magnetrun.analysis.field_comparison.compare_all_fields` runs
   internally, without needing an ``OverviewRecord`` backed by real files.
3. Benchmark the two lag algorithms in
   :mod:`python_magnetrun.analysis.synchronization`: ``compute_lag`` (fixed
   1 s resample) vs. ``compute_lag_interpolated`` (common fine grid), for
   both an Overview-like (1 Hz) and an Archive-like (120 Hz) scenario
   against irregularly-sampled pupitre data.

Usage
-----
::

    python field_comparison_demo.py
    python field_comparison_demo.py --repeat 10 --seed 7

    # With plots (off by default so the script stays fast/non-interactive)
    python field_comparison_demo.py --show
    python field_comparison_demo.py --output-dir ./plots
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tabulate import tabulate

from python_magnetrun.analysis.field_comparison import (
    AliasedField,
    compare_field,
    compute_reference_lag,
    discover_pupitre_pigbrother_fields,
    print_comparison_summary,
)
from python_magnetrun.analysis.synchronization import (
    apply_lag_correction,
    compute_lag,
    compute_lag_interpolated,
)

# Same stream as print() so PART 1's log lines interleave in call order
# instead of racing stdout against a separate stderr buffer.
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

ORIGIN = pd.Timestamp("2024-01-01 00:00:00")

#: Lag injected into every synthetic "pupitre" column below [s]. Deliberately
#: sub-second/non-integer to make the precision difference between the two
#: lag algorithms visible.
TRUE_LAG = 2.3


def _bump_df(
    t: np.ndarray, columns_to_lag: dict[str, tuple[float, float]], width: float = 1.0
) -> pd.DataFrame:
    """Build a DataFrame with one Gaussian-bump column per entry in *columns_to_lag*.

    ``columns_to_lag[name] = (amplitude, lag)`` places a bump for *name*
    centered at ``t = center + lag`` (*center* is the midpoint of *t*), so a
    "pupitre" column built with ``lag=TRUE_LAG`` is a delayed copy of a
    "pigbrother" column built with ``lag=0``. *width* is the bump's standard
    deviation [s] and should scale with the window length -- a bump that is
    too narrow relative to a long, near-flat window makes unnormalized
    cross-correlation dominated by that flat background rather than the
    feature (keep the window-to-width ratio around 20:1, as in this module's
    scenarios).
    """
    center = t[len(t) // 2]
    data: dict[str, object] = {"timestamp": ORIGIN + pd.to_timedelta(t, unit="s")}
    for name, (amplitude, lag) in columns_to_lag.items():
        data[name] = amplitude * np.exp(-((t - center - lag) ** 2) / (2 * width**2))
    return pd.DataFrame(data)


def _irregular_times(duration: float, mean_step: float, seed: int) -> np.ndarray:
    """Irregularly-spaced timestamps (seconds) spanning ``[0, duration)``."""
    rng = np.random.default_rng(seed)
    n = int(duration / mean_step) + 20
    steps = mean_step + rng.uniform(-0.4 * mean_step, 0.4 * mean_step, size=n)
    t = np.cumsum(steps)
    return t[t < duration]


# ---------------------------------------------------------------------------
# Part 1: demo -- discovery + one reference-lag-driven comparison
# ---------------------------------------------------------------------------
class _FakeRecord:
    """Stand-in for OverviewRecord: compute_reference_lag only uses .filename."""

    filename = "demo"


def run_demo(show: bool = False, output_dir: str | None = None) -> None:
    plot_enabled = show or output_dir is not None

    print("=" * 70)
    print("PART 1 -- field discovery + reference-lag-driven comparison")
    print("=" * 70)

    fields = discover_pupitre_pigbrother_fields()
    print(f"\nDiscovered {len(fields)} pupitre <-> pigbrother aliased fields, e.g.:")
    for f in fields[:5]:
        print(f"  {f.pupitre_key:10s} <-> {f.pigbrother_group}/{f.pigbrother_channel}")

    # Overview-like data: 1 Hz over 1 minute; pupitre is delayed by TRUE_LAG.
    # Bump width (3 s) keeps a 20:1 window-to-width ratio -- see _bump_df.
    t = np.arange(0, 60, 1.0)
    pupitre_df = _bump_df(
        t,
        {
            "Idcct1": (1.0, TRUE_LAG),
            "Idcct3": (1.0, TRUE_LAG),
            "Ucoil1": (2.0, TRUE_LAG),
        },
        width=3.0,
    )
    overview_currents = _bump_df(t, {"Courant_A1": (1.0, 0.0)}, width=3.0)
    overview_voltages = _bump_df(t, {"Interne1": (2.0, 0.0)}, width=3.0)

    reference_fields = {
        "Idcct1": AliasedField("Idcct1", "Courants_Alimentations", "Courant_A1"),
        "Idcct3": AliasedField("Idcct3", "Courants_Alimentations", "Courant_A3"),
    }
    reference_lag, reference_field = compute_reference_lag(
        _FakeRecord(), "overview", pupitre_df, overview_currents, reference_fields
    )
    print(
        f"\nReference lag ({reference_field.pupitre_key} vs "
        f"{reference_field.pigbrother_channel}): {reference_lag.lag.total_seconds():.3f} s"
    )

    pupitre_corrected = apply_lag_correction(pupitre_df, reference_lag.lag)

    idcct1_result = compare_field(
        AliasedField("Idcct1", "Courants_Alimentations", "Courant_A1"),
        "overview",
        overview_currents,
        pupitre_corrected,
        plot=plot_enabled,
        output_dir=output_dir,
        show=show,
    )
    idcct1_result.reference_lag = reference_lag
    idcct1_result.reference_field = reference_field

    ucoil1_result = compare_field(
        AliasedField("Ucoil1", "Tensions_Aimant", "Interne1"),
        "overview",
        overview_voltages,
        pupitre_corrected,
        plot=plot_enabled,
        output_dir=output_dir,
        show=show,
    )
    ucoil1_result.reference_lag = reference_lag
    ucoil1_result.reference_field = reference_field

    results = {
        "Idcct1": {"overview": idcct1_result},
        "Ucoil1": {"overview": ucoil1_result},
    }
    print()
    print_comparison_summary(results)
    if plot_enabled and output_dir:
        print(f"Saved comparison plots to {output_dir}/")


# ---------------------------------------------------------------------------
# Part 2: benchmark -- compute_lag vs. compute_lag_interpolated
# ---------------------------------------------------------------------------
@dataclass
class _BenchRow:
    scenario: str
    method: str
    time_mean_ms: float
    lag_seconds: float
    abs_error_s: float


def _time_calls(fn, repeat: int) -> tuple[object, float]:
    """Call *fn* *repeat* times; return (last_result, mean_time_ms)."""
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, float(np.mean(times)) * 1000.0


def _benchmark_scenario(
    scenario: str, dt1: float, duration: float, width: float, seed: int, repeat: int
) -> list[_BenchRow]:
    t1 = np.arange(0, duration, dt1)
    df1 = _bump_df(t1, {"value": (1.0, 0.0)}, width=width)
    t2 = _irregular_times(duration, mean_step=0.4, seed=seed)
    df2 = _bump_df(t2, {"value": (1.0, TRUE_LAG)}, width=width)

    df1_data = {
        "df": df1[["timestamp", "value"]],
        "field": "value",
        "range": {"start": 0, "end": None},
    }
    df2_data = {
        "df": df2[["timestamp", "value"]],
        "field": "value",
        "range": {"start": 0, "end": None},
    }

    old_result, old_ms = _time_calls(lambda: compute_lag("timestamp", df1_data, df2_data), repeat)
    new_result, new_ms = _time_calls(
        lambda: compute_lag_interpolated("timestamp", df1_data, df2_data), repeat
    )

    old_lag = old_result.total_seconds()
    new_lag = new_result.lag.total_seconds()

    return [
        _BenchRow(scenario, "resample_1s (compute_lag)", old_ms, old_lag, abs(old_lag - TRUE_LAG)),
        _BenchRow(scenario, "interpolated", new_ms, new_lag, abs(new_lag - TRUE_LAG)),
    ]


# Fixed method -> color assignment (Okabe-Ito colorblind-safe pair), used
# consistently across both subplots -- never reassigned or cycled.
_METHOD_COLORS = {
    "resample_1s (compute_lag)": "#0072B2",
    "interpolated": "#E69F00",
}


def _plot_benchmark(rows: list[_BenchRow], show: bool, output_dir: str | None) -> None:
    """Grouped bar charts: time and accuracy, one measure per axis (no dual-axis)."""
    if not show and not output_dir:
        return

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(r.scenario for r in rows))
    methods = list(_METHOD_COLORS)
    x = np.arange(len(scenarios))
    bar_width = 0.35

    fig, (ax_time, ax_error) = plt.subplots(1, 2, figsize=(11, 4.5))

    for i, method in enumerate(methods):
        offset = (i - 0.5) * bar_width
        times = [
            next(r.time_mean_ms for r in rows if r.scenario == s and r.method == method)
            for s in scenarios
        ]
        errors = [
            next(r.abs_error_s for r in rows if r.scenario == s and r.method == method)
            for s in scenarios
        ]
        ax_time.bar(x + offset, times, bar_width, label=method, color=_METHOD_COLORS[method])
        ax_error.bar(x + offset, errors, bar_width, label=method, color=_METHOD_COLORS[method])

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(scenarios, rotation=10, ha="right")
    ax_time.set_ylabel("Mean time per call (ms)")
    ax_time.set_title("Computation time")
    ax_time.legend()
    ax_time.grid(axis="y", alpha=0.3)

    ax_error.set_xticks(x)
    ax_error.set_xticklabels(scenarios, rotation=10, ha="right")
    ax_error.set_ylabel(f"|recovered lag - true lag| (s); true={TRUE_LAG} s")
    ax_error.set_yscale("log")
    ax_error.set_title("Accuracy")
    ax_error.legend()
    ax_error.grid(axis="y", alpha=0.3, which="both")

    fig.suptitle("compute_lag vs. compute_lag_interpolated")
    fig.tight_layout()

    if output_dir:
        path = f"{output_dir}/lag_benchmark.png"
        fig.savefig(path, dpi=150)
        print(f"Saved benchmark plot to {path}")
    if show:
        plt.show()
    plt.close(fig)


def run_benchmark(
    repeat: int, seed: int, show: bool = False, output_dir: str | None = None
) -> None:
    print("\n" + "=" * 70)
    print("PART 2 -- benchmark: compute_lag vs. compute_lag_interpolated")
    print("=" * 70)
    print(f"\nTrue injected lag: {TRUE_LAG} s (repeat={repeat}, seed={seed})\n")

    # Window-to-width kept at ~20:1 in both scenarios (see _bump_df) so the
    # cross-correlation has an unambiguous peak instead of being dominated by
    # a long near-flat background.
    rows = _benchmark_scenario(
        "Overview-like (1 Hz) vs pupitre",
        dt1=1.0,
        duration=60.0,
        width=3.0,
        seed=seed,
        repeat=repeat,
    ) + _benchmark_scenario(
        "Archive-like (120 Hz) vs pupitre",
        dt1=1.0 / 120.0,
        duration=10.0,
        width=0.5,
        seed=seed,
        repeat=repeat,
    )

    table = [
        [r.scenario, r.method, f"{r.time_mean_ms:.3f}", f"{r.lag_seconds:.3f}", f"{r.abs_error_s:.3f}"]
        for r in rows
    ]
    print(
        tabulate(
            table,
            headers=["Scenario", "Method", "Time (ms)", "Recovered lag (s)", "Abs. error (s)"],
            tablefmt="simple",
        )
    )

    _plot_benchmark(rows, show=show, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat", type=int, default=5, help="timing repetitions per method (default: 5)"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for irregular sampling")
    parser.add_argument(
        "--show", action="store_true", help="display plots interactively (requires a display)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory to save plots (PNG) to; created if missing",
    )
    args = parser.parse_args()

    if args.output_dir:
        from pathlib import Path

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_demo(show=args.show, output_dir=args.output_dir)
    run_benchmark(repeat=args.repeat, seed=args.seed, show=args.show, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
