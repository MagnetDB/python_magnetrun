#!/usr/bin/env python3
"""
Example: downsampling a large DataFrame before plotting.

Demonstrates the three main entry-points in ``python_magnetrun.utils.downsampling``:

- ``DownsampleConfig``          — configuration dataclass
- ``downsample_arrays``         — for (data, time) numpy array pairs
- ``downsample_dataframe``      — for multi-column DataFrames

The script generates a synthetic 100 000-point time series (two channels: a
slow sinusoid and high-frequency noise), applies several downsampling methods,
and plots the results side-by-side with matplotlib.

Usage::

    python downsampling_example.py                 # default: stride, 2000 pts
    python downsampling_example.py --method lttb --n-out 1000
    python downsampling_example.py --method minmax_lttb --n-out 500
    python downsampling_example.py --percent 2.0   # 2% of dataset length
    python downsampling_example.py --no-plot       # just print statistics
"""

from __future__ import annotations

import argparse
import sys
import time as _time

import numpy as np
import pandas as pd

from python_magnetrun.utils.downsampling import (
    DownsampleConfig,
    downsample_arrays,
    downsample_dataframe,
)

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def make_large_dataframe(n: int = 100_000) -> pd.DataFrame:
    """Return a DataFrame with *n* rows and three value columns."""
    t = np.linspace(0.0, 3600.0, n)  # 1-hour run at 1 ms resolution
    slow = 10.0 * np.sin(2 * np.pi * t / 600)  # 10-min period
    fast = 0.5 * np.sin(2 * np.pi * t / 0.5) + 0.1 * np.random.default_rng(
        0
    ).standard_normal(n)
    current = (
        12_000
        + 500 * np.sin(2 * np.pi * t / 1200)
        + 20 * np.random.default_rng(1).standard_normal(n)
    )
    return pd.DataFrame({"t": t, "Bfield": slow, "noise_ch": fast, "Icoil": current})


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def demo_arrays(df: pd.DataFrame, config: DownsampleConfig) -> None:
    """Demonstrate ``downsample_arrays`` on a single column."""
    print("\n--- downsample_arrays ---")
    time_arr = df["t"].to_numpy(dtype=float)
    data_arr = df["Icoil"].to_numpy(dtype=float)

    t0 = _time.perf_counter()
    ds_data, ds_time = downsample_arrays(data_arr, time_arr, config)
    elapsed = _time.perf_counter() - t0

    print(f"  Input  : {len(data_arr):>8d} points")
    print(
        f"  Output : {len(ds_data):>8d} points  ({len(ds_data)/len(data_arr)*100:.1f} %)"
    )
    print(f"  Method : {config.method}")
    print(f"  Time   : {elapsed*1000:.1f} ms")


def demo_dataframe(df: pd.DataFrame, config: DownsampleConfig) -> pd.DataFrame:
    """Demonstrate ``downsample_dataframe`` on the whole DataFrame."""
    print("\n--- downsample_dataframe ---")

    t0 = _time.perf_counter()
    ds_df = downsample_dataframe(
        df,
        time_col="t",
        value_cols=["Bfield", "Icoil"],
        config=config,
    )
    elapsed = _time.perf_counter() - t0

    print(f"  Input  : {len(df):>8d} rows × {len(df.columns)} cols")
    print(
        f"  Output : {len(ds_df):>8d} rows × {len(ds_df.columns)} cols  ({len(ds_df)/len(df)*100:.1f} %)"
    )
    print(f"  Method : {config.method}")
    print(f"  Time   : {elapsed*1000:.1f} ms")
    print(f"  Stored config in df.attrs: {ds_df.attrs.get('downsample_config')}")
    return ds_df


def demo_from_percent(df: pd.DataFrame, percent: float) -> None:
    """Demonstrate ``DownsampleConfig.from_percent`` (bridges analysis/ model)."""
    print("\n--- DownsampleConfig.from_percent ---")
    config = DownsampleConfig.from_percent(len(df), percent=percent, method="stride")
    print(f"  {percent:.1f}% of {len(df)} → n_out={config.n_out}")
    demo_dataframe(df, config)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(
    df: pd.DataFrame, ds_df: pd.DataFrame, config: DownsampleConfig
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    fig.suptitle(
        f"Downsampling demo  |  method={config.method}  |  "
        f"{len(df):,} → {len(ds_df):,} points",
        fontsize=12,
    )

    # --- Icoil full-range comparison ---
    ax = axes[0, 0]
    ax.plot(df["t"], df["Icoil"], lw=0.3, alpha=0.4, label="original")
    ax.plot(ds_df["t"], ds_df["Icoil"], lw=0.8, label=f"downsampled ({config.method})")
    ax.set_title("Icoil — full run")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Icoil (A)")
    ax.legend(fontsize=8)

    # --- Bfield full-range comparison ---
    ax = axes[0, 1]
    ax.plot(df["t"], df["Bfield"], lw=0.3, alpha=0.4, label="original")
    ax.plot(ds_df["t"], ds_df["Bfield"], lw=0.8, label=f"downsampled ({config.method})")
    ax.set_title("Bfield — full run")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("B (T)")
    ax.legend(fontsize=8)

    # --- Icoil zoomed (first 60 s) ---
    ax = axes[1, 0]
    mask_full = df["t"] < 60
    mask_ds = ds_df["t"] < 60
    ax.plot(
        df.loc[mask_full, "t"],
        df.loc[mask_full, "Icoil"],
        lw=0.5,
        alpha=0.5,
        label="original",
    )
    ax.plot(
        ds_df.loc[mask_ds, "t"],
        ds_df.loc[mask_ds, "Icoil"],
        "o-",
        ms=2,
        lw=0.8,
        label="downsampled",
    )
    ax.set_title("Icoil — first 60 s (zoom)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Icoil (A)")
    ax.legend(fontsize=8)

    # --- noise_ch zoomed (first 5 s — high-frequency signal) ---
    ax = axes[1, 1]
    mask_full = df["t"] < 5
    mask_ds = ds_df["t"] < 5
    ax.plot(
        df.loc[mask_full, "t"],
        df.loc[mask_full, "noise_ch"],
        lw=0.4,
        alpha=0.5,
        label="original",
    )
    ax.plot(
        ds_df.loc[mask_ds, "t"],
        ds_df.loc[mask_ds, "noise_ch"],
        "o-",
        ms=2,
        lw=0.8,
        label="downsampled",
    )
    ax.set_title("noise_ch — first 5 s (high-freq zoom)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("noise_ch")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = f"downsampling_{config.method}_{config.n_out}.png"
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n-rows",
        type=int,
        default=100_000,
        metavar="N",
        help="synthetic dataset size (default: 100 000)",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--n-out",
        type=int,
        default=2_000,
        metavar="N",
        help="target output points (default: 2 000)",
    )
    grp.add_argument(
        "--percent",
        type=float,
        metavar="PCT",
        help="target size as %% of dataset (e.g. 2.0 → 2%%)",
    )
    p.add_argument(
        "--method",
        default="stride",
        choices=["stride", "minmax", "lttb", "minmax_lttb"],
        help="downsampling algorithm (default: stride)",
    )
    p.add_argument("--no-plot", action="store_true", help="skip matplotlib plot")
    return p


def main() -> int:
    args = build_parser().parse_args()

    print(f"Generating synthetic DataFrame ({args.n_rows:,} rows) …")
    df = make_large_dataframe(args.n_rows)
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Build config
    if args.percent is not None:
        config = DownsampleConfig.from_percent(
            len(df), percent=args.percent, method=args.method
        )
        print(f"\nConfig (from_percent): {config}")
    else:
        config = DownsampleConfig(n_out=args.n_out, method=args.method)
        print(f"\nConfig: {config}")

    # Demo array path
    demo_arrays(df, config)

    # Demo DataFrame path (returns downsampled copy)
    ds_df = demo_dataframe(df, config)

    # Demo from_percent bridge
    demo_from_percent(df, percent=1.0)

    # Plot
    if not args.no_plot:
        plot_comparison(df, ds_df, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
