#!/usr/bin/env python3
"""
Example: downsampling a large DataFrame with Plotly and plotly-resampler.

Two complementary strategies are demonstrated:

1. **Plotly + pre-computed downsampling** — Run ``DownsampleConfig`` /
   ``downsample_dataframe`` before plotting, then render with
   ``PlotlyBackend``.  Output is a self-contained HTML file (interactive) or
   a static PNG/SVG with kaleido.  No live kernel required.

2. **plotly-resampler — dynamic, view-dependent downsampling** — Feed the
   full raw ``DataFrame`` directly to ``PlotlyResamplerBackend``.
   ``FigureResampler`` (Dash) or ``FigureWidgetResampler`` (Jupyter/marimo)
   re-aggregates via MinMaxLTTB on every pan/zoom.  Requires a live Python
   kernel — static export is not supported.

Usage::

    # Plotly backend — pre-computed stride (default)
    python plotly_downsampling_example.py

    # Plotly backend — different algorithm or target size
    python plotly_downsampling_example.py --method lttb --n-out 1000
    python plotly_downsampling_example.py --method minmax_lttb --percent 1.0

    # plotly-resampler — Dash app (opens browser)
    python plotly_downsampling_example.py --backend resampler

    # plotly-resampler — Jupyter/marimo widget mode
    python plotly_downsampling_example.py --backend resampler --widget

    # Just save HTML, skip browser / Dash server
    python plotly_downsampling_example.py --no-show
"""

from __future__ import annotations

import argparse
import sys
import time as _time

import numpy as np
import pandas as pd

from python_magnetrun.utils.downsampling import (
    DownsampleConfig,
    downsample_dataframe,
)

# ---------------------------------------------------------------------------
# Synthetic data generator  (same as downsampling_example.py)
# ---------------------------------------------------------------------------


def make_large_dataframe(n: int = 100_000) -> pd.DataFrame:
    """Return a DataFrame with *n* rows and three value columns."""
    t = np.linspace(0.0, 3600.0, n)
    slow = 10.0 * np.sin(2 * np.pi * t / 600)
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
# Strategy 1 — PlotlyBackend + pre-computed downsampling
# ---------------------------------------------------------------------------


def plot_plotly_precomputed(
    df: pd.DataFrame,
    config: DownsampleConfig,
    *,
    show: bool = True,
) -> None:
    """Pre-downsample *df*, build a 4-row Plotly figure, save as HTML."""
    try:
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend
    except ImportError as exc:
        print(f"PlotlyBackend unavailable: {exc}")
        return

    print("\n=== Strategy 1: PlotlyBackend + pre-computed downsampling ===")

    # Downsample before plotting
    t0 = _time.perf_counter()
    ds_df = downsample_dataframe(
        df,
        time_col="t",
        value_cols=["Bfield", "Icoil", "noise_ch"],
        config=config,
    )
    elapsed = _time.perf_counter() - t0
    print(
        f"  Downsampled: {len(df):,} → {len(ds_df):,} rows "
        f"({len(ds_df) / len(df) * 100:.1f} %)  [{elapsed * 1000:.1f} ms]"
    )

    backend = PlotlyBackend()
    fig = backend.subplots(4, share_x=False)
    fig.update_layout(
        title_text=(
            f"Plotly — pre-computed downsampling  |  method={config.method}  |  "
            f"{len(df):,} → {len(ds_df):,} pts"
        )
    )

    # Row 0 — Icoil full run
    backend.add_series(
        fig, 0,
        df["t"].to_numpy(), df["Icoil"].to_numpy(),
        "Icoil original", alpha=0.3, ylabel="Icoil (A)",
    )
    backend.add_series(
        fig, 0,
        ds_df["t"].to_numpy(), ds_df["Icoil"].to_numpy(),
        f"Icoil ({config.method})", ylabel="Icoil (A)",
    )

    # Row 1 — Bfield full run
    backend.add_series(
        fig, 1,
        df["t"].to_numpy(), df["Bfield"].to_numpy(),
        "Bfield original", alpha=0.3, ylabel="B (T)",
    )
    backend.add_series(
        fig, 1,
        ds_df["t"].to_numpy(), ds_df["Bfield"].to_numpy(),
        f"Bfield ({config.method})", ylabel="B (T)",
    )

    # Row 2 — Icoil zoomed (first 60 s)
    mask_full = df["t"] < 60
    mask_ds = ds_df["t"] < 60
    backend.add_series(
        fig, 2,
        df.loc[mask_full, "t"].to_numpy(), df.loc[mask_full, "Icoil"].to_numpy(),
        "Icoil original (zoom)", alpha=0.4, ylabel="Icoil (A)",
    )
    backend.add_series(
        fig, 2,
        ds_df.loc[mask_ds, "t"].to_numpy(), ds_df.loc[mask_ds, "Icoil"].to_numpy(),
        "Icoil downsampled (zoom)", marker="o", markevery=1, ylabel="Icoil (A)",
    )

    # Row 3 — noise_ch zoomed (first 5 s, high-frequency)
    mask_full = df["t"] < 5
    mask_ds = ds_df["t"] < 5
    backend.add_series(
        fig, 3,
        df.loc[mask_full, "t"].to_numpy(), df.loc[mask_full, "noise_ch"].to_numpy(),
        "noise_ch original (zoom)", alpha=0.4, ylabel="noise",
    )
    backend.add_series(
        fig, 3,
        ds_df.loc[mask_ds, "t"].to_numpy(), ds_df.loc[mask_ds, "noise_ch"].to_numpy(),
        "noise_ch downsampled (zoom)", marker="o", markevery=1, ylabel="noise",
    )

    fig.update_xaxes(title_text="t (s)", row=4, col=1)

    out_html = f"plotly_precomputed_{config.method}_{config.n_out}.html"
    from pathlib import Path
    backend.save(fig, Path(out_html))
    print(f"  Saved → {out_html}")

    if show:
        backend.show(fig)


# ---------------------------------------------------------------------------
# Strategy 2 — PlotlyResamplerBackend + raw data (dynamic resampling)
# ---------------------------------------------------------------------------


def plot_plotly_resampler(
    df: pd.DataFrame,
    *,
    widget: bool = False,
    show: bool = True,
) -> None:
    """Feed *full* raw DataFrame to PlotlyResamplerBackend (dynamic resampling)."""
    try:
        from python_magnetrun.plotting.plotly_resampler_backend import (
            PlotlyResamplerBackend,
        )
    except ImportError as exc:
        print(f"PlotlyResamplerBackend unavailable: {exc}")
        return

    mode = "widget (Jupyter/marimo)" if widget else "Dash"
    print(f"\n=== Strategy 2: PlotlyResamplerBackend — {mode} ===")
    print(f"  Feeding {len(df):,} raw rows — resampling is view-dependent.")

    try:
        backend = PlotlyResamplerBackend(widget=widget)
        fig = backend.subplots(4, share_x=False)
    except ImportError as exc:
        print(f"  plotly-resampler not installed: {exc}")
        print("  Install with: pip install python_magnetrun[resampler]")
        return

    fig.update_layout(
        title_text=(
            f"plotly-resampler — dynamic downsampling  |  {mode}  |  "
            f"{len(df):,} raw pts (MinMaxLTTB on pan/zoom)"
        )
    )

    # All four rows receive the *full* arrays — FigureResampler aggregates lazily.
    backend.add_series(
        fig, 0,
        df["t"].to_numpy(), df["Icoil"].to_numpy(),
        "Icoil", ylabel="Icoil (A)",
    )
    backend.add_series(
        fig, 1,
        df["t"].to_numpy(), df["Bfield"].to_numpy(),
        "Bfield", ylabel="B (T)",
    )
    backend.add_series(
        fig, 2,
        df["t"].to_numpy(), df["noise_ch"].to_numpy(),
        "noise_ch", ylabel="noise",
    )
    # Overlay Icoil + Bfield normalised on the same panel
    backend.add_series(
        fig, 3,
        df["t"].to_numpy(), df["Icoil"].to_numpy(),
        "Icoil (norm)", normalize=True, ylabel="normalised",
    )
    backend.add_series(
        fig, 3,
        df["t"].to_numpy(), df["Bfield"].to_numpy(),
        "Bfield (norm)", normalize=True,
    )

    fig.update_xaxes(title_text="t (s)", row=4, col=1)

    if show:
        print("  Launching…  (Ctrl-C to stop)")
        backend.show(fig)
    else:
        print(
            "  --no-show: skipped.  "
            "Call backend.show(fig) / fig.show_dash() in an interactive session."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_resampler_hint() -> None:
    """Print a one-line hint about Strategy 2 / plotly-resampler availability."""
    try:
        from plotly_resampler import FigureResampler  # noqa: F401
        available = True
    except ImportError:
        available = False

    print()
    if available:
        print(
            "Tip — Strategy 2 (plotly-resampler) is installed.\n"
            "  Run with --backend resampler for dynamic, view-dependent downsampling."
        )
    else:
        print(
            "Tip — Strategy 2 (plotly-resampler) is NOT installed.\n"
            "  Install with: pip install python_magnetrun[resampler]\n"
            "  Then run with: --backend resampler"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n-rows", type=int, default=100_000, metavar="N",
        help="synthetic dataset size (default: 100 000)",
    )
    p.add_argument(
        "--backend", default="plotly", choices=["plotly", "resampler"],
        help="plotting backend (default: plotly)",
    )
    p.add_argument(
        "--widget", action="store_true",
        help="[resampler only] use FigureWidgetResampler (Jupyter/marimo)",
    )

    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--n-out", type=int, default=2_000, metavar="N",
        help="[plotly only] target output points (default: 2 000)",
    )
    grp.add_argument(
        "--percent", type=float, metavar="PCT",
        help="[plotly only] target size as %% of dataset (e.g. 1.0 → 1%%)",
    )
    p.add_argument(
        "--method", default="stride",
        choices=["stride", "minmax", "lttb", "minmax_lttb"],
        help="[plotly only] downsampling algorithm (default: stride)",
    )
    p.add_argument(
        "--no-show", dest="show", action="store_false",
        help="save HTML (plotly) or skip show (resampler) without opening browser",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    print(f"Generating synthetic DataFrame ({args.n_rows:,} rows) …")
    df = make_large_dataframe(args.n_rows)
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    if args.backend == "plotly":
        if args.percent is not None:
            config = DownsampleConfig.from_percent(
                len(df), percent=args.percent, method=args.method
            )
        else:
            config = DownsampleConfig(n_out=args.n_out, method=args.method)
        print(f"\nDownsample config: {config}")
        plot_plotly_precomputed(df, config, show=args.show)
        _print_resampler_hint()
    else:
        plot_plotly_resampler(df, widget=args.widget, show=args.show)

    return 0


if __name__ == "__main__":
    sys.exit(main())
