#!/usr/bin/env python3
"""Benchmark magnet data loading across file types.

Resolves input files via :func:`~python_magnetrun.utils.files.expand_input_files`,
discovers related files for each TDMS overview via
:class:`~python_magnetrun.analysis.loaders.FileDiscovery`, then times
:func:`~python_magnetrun.MagnetRun.load_mrun` per file and reports statistics
and figures.

Usage::

    # Benchmark files discovered from an overview TDMS
    python benchmark_loading.py M8_Overview_25*.tdms --housing M8 \\
        --pigbrother_datadir /data/pbsurv --pupitre_datadir /data/pupitre --show

    # Benchmark a mix of overview TDMS and standalone pupitre files
    python benchmark_loading.py M8_Overview_251105-0949.tdms 2025*.txt \\
        --housing M8 --pigbrother_datadir data --show

    # Save figure and raw CSV
    python benchmark_loading.py M8_Overview_25*.tdms --housing M8 --save \\
        --output-dir results --repeat 3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark_loading")


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class _FileResult:
    """Timing result for one file.

    Attributes
    ----------
    filepath : str
        Absolute or resolved path to the file.
    category : str
        FileSet category label (``"overview"``, ``"archive"``, ``"pupitre"``, …).
    housing : str
        Housing identifier used for loading [dimensionless].
    size_bytes : int
        File size [bytes].
    n_rows : int
        Row count returned by the loader.
    time_min_s : float
        Minimum wall-clock load time across repetitions [s].
    time_mean_s : float
        Mean wall-clock load time across repetitions [s].
    error : str
        Non-empty when the load failed.
    """

    filepath: str
    category: str
    housing: str
    size_bytes: int = 0
    n_rows: int = 0
    time_min_s: float = float("nan")
    time_mean_s: float = float("nan")
    error: str = ""

    @property
    def ext(self) -> str:
        """File extension (lower-case)."""
        return Path(self.filepath).suffix.lower()

    @property
    def size_mb(self) -> float:
        """File size [MB]."""
        return self.size_bytes / 1_048_576


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _collect_files(
    input_files: list[str],
    housing: str,
    pigbrother_datadir: str,
    pupitre_datadir: str,
    discovery: bool = True,
) -> list[tuple[str, str]]:
    """Return ``(category, filepath)`` pairs for all files to benchmark.

    Parameters
    ----------
    input_files : list of str
        Resolved file paths (output of :func:`expand_input_files`).
    housing : str
        Housing identifier forwarded to :class:`FileDiscovery`.
    pigbrother_datadir : str
        Root directory for TDMS files.
    pupitre_datadir : str
        Root directory for pupitre ``.txt`` files.
    discovery : bool
        When ``True`` (default), call :class:`FileDiscovery` on each ``.tdms``
        input to collect related archive, pupitre, and incident files.
        When ``False``, benchmark only the input files themselves.

    Returns
    -------
    list of (str, str)
        ``(category_label, filepath)`` pairs, deduplicated.
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []

    def _add(category: str, path: str) -> None:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            pairs.append((category, abs_path))

    if discovery:
        from python_magnetrun.analysis.loaders import FileDiscovery

        file_discovery = FileDiscovery(
            pupitre_datadir=pupitre_datadir,
            pigbrother_datadir=pigbrother_datadir,
        )

    for path in input_files:
        ext = Path(path).suffix.lower()
        if ext == ".tdms":
            if discovery:
                file_set = file_discovery.discover(
                    path,
                    housing=housing if housing not in ("notdefined", "") else None,
                )
                for category in (
                    "overview", "archive", "pupitre",
                    "default", "trigger", "spike",
                    "hybrid_kHz", "hybrid_rms", "hybrid_trigger", "hybrid_vprocess",
                ):
                    for fpath in getattr(file_set, category, []):
                        _add(category, fpath)
            else:
                _add("tdms", path)
        else:
            category = "pupitre" if ext == ".txt" else "csv"
            _add(category, path)

    return pairs


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_load(filepath: str, housing: str, repeat: int) -> tuple[int, float, float]:
    """Load *filepath* *repeat* times and return ``(n_rows, time_min_s, time_mean_s)``.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    housing : str
        Housing identifier [dimensionless].
    repeat : int
        Number of load repetitions (minimum 1).

    Returns
    -------
    tuple of (int, float, float)
        ``(n_rows, time_min_s, time_mean_s)``
    """
    from python_magnetrun.MagnetRun import load_mrun

    times: list[float] = []
    n_rows = 0

    for i in range(max(repeat, 1)):
        t0 = time.perf_counter()
        mrun = load_mrun(filepath, housing=housing, auto_resolve=False)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if i == 0:
            try:
                mdata = mrun.getMData()
                keys = mdata.getKeys()
                if keys:
                    n_rows = len(mdata.getData(keys[0]))
            except Exception:  # noqa: BLE001
                n_rows = 0

    return n_rows, min(times), sum(times) / len(times)


def _benchmark(
    pairs: list[tuple[str, str]],
    housing: str,
    repeat: int,
) -> list[_FileResult]:
    """Benchmark each ``(category, filepath)`` pair.

    Parameters
    ----------
    pairs : list of (str, str)
        ``(category, filepath)`` pairs from :func:`_collect_files`.
    housing : str
        Housing identifier [dimensionless].
    repeat : int
        Number of timing repetitions per file.

    Returns
    -------
    list of _FileResult
        One record per file.
    """
    results: list[_FileResult] = []

    for category, filepath in pairs:
        result = _FileResult(filepath=filepath, category=category, housing=housing)

        if not os.path.isfile(filepath):
            result.error = "file not found"
            results.append(result)
            logger.warning("skipping %s: file not found", filepath)
            continue

        result.size_bytes = os.path.getsize(filepath)

        try:
            n_rows, t_min, t_mean = _time_load(filepath, housing, repeat)
            result.n_rows = n_rows
            result.time_min_s = t_min
            result.time_mean_s = t_mean
            logger.info(
                "%s [%s] %.3f s (min), %.3f s (mean), %d rows, %.2f MB",
                filepath, category, t_min, t_mean, n_rows, result.size_mb,
            )
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            logger.error("failed to load %s: %s", filepath, exc)

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _print_stats(df: pd.DataFrame) -> None:
    """Print a per-category summary table to stdout.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw results DataFrame with columns ``category``, ``size_mb``,
        ``time_min_s``, ``time_mean_s``, ``n_rows``.
    """
    ok = df[df["error"] == ""]
    if ok.empty:
        print("No files loaded successfully.")
        return

    agg = ok.groupby("category").agg(
        count=("filepath", "count"),
        size_mb_mean=("size_mb", "mean"),
        size_mb_max=("size_mb", "max"),
        time_min_s=("time_min_s", "min"),
        time_mean_s=("time_mean_s", "mean"),
        time_max_s=("time_min_s", "max"),
        rows_median=("n_rows", "median"),
    )
    print("\n=== Benchmark results by category ===")
    print(agg.to_string(float_format="{:.3f}".format))
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(df: pd.DataFrame, output_dir: Path, save: bool, show: bool, title: str | None) -> None:
    """Produce and optionally save / display benchmark figures.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw results DataFrame.
    output_dir : Path
        Directory for saved figure.
    save : bool
        When ``True``, write ``benchmark_loading.png`` to *output_dir*.
    show : bool
        When ``True``, call :func:`matplotlib.pyplot.show`.
    title : str or None
        Optional figure suptitle override.
    """
    import matplotlib.pyplot as plt

    ok = df[df["error"] == ""]
    if ok.empty:
        logger.warning("no successful results to plot")
        return

    categories = sorted(ok["category"].unique())
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    cat_color = {c: colors[i % len(colors)] for i, c in enumerate(categories)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title or "Magnet data loading benchmark")

    # --- box plot: load time per category ---
    ax = axes[0]
    data_by_cat = [ok.loc[ok["category"] == c, "time_min_s"].dropna().values for c in categories]
    bp = ax.boxplot(data_by_cat, labels=categories, patch_artist=True)
    for patch, cat in zip(bp["boxes"], categories, strict=False):
        patch.set_facecolor(cat_color[cat])
    ax.set_xlabel("Category")
    ax.set_ylabel("Load time [s]")
    ax.set_title("Load time per category")
    ax.tick_params(axis="x", rotation=30)

    # --- scatter: load time vs file size ---
    ax = axes[1]
    for cat in categories:
        sub = ok[ok["category"] == cat]
        ax.scatter(sub["size_mb"], sub["time_min_s"], label=cat, color=cat_color[cat], alpha=0.7)
    ax.set_xlabel("File size [MB]")
    ax.set_ylabel("Load time [s]")
    ax.set_title("Load time vs file size")
    ax.legend(fontsize="small")

    # --- bar: median row count per category ---
    ax = axes[2]
    medians = ok.groupby("category")["n_rows"].median().reindex(categories)
    bar_colors = [cat_color[c] for c in categories]
    ax.bar(categories, medians.values, color=bar_colors)
    ax.set_xlabel("Category")
    ax.set_ylabel("Row count (median)")
    ax.set_title("Row count per category")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "benchmark_loading.png"
        fig.savefig(out_path, dpi=150)
        print(f"Figure saved: {out_path}")

    if show:
        plt.show()
    elif not save:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the benchmark script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    from python_magnetrun.cli_args import create_base_parser, create_managed_plots_parser

    base = create_base_parser([".tdms", ".txt", ".csv"])
    managed = create_managed_plots_parser()

    parser = argparse.ArgumentParser(
        description="Benchmark magnet data loading by file type.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[base, managed],
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        default=False,
        help="benchmark only the input files themselves; skip FileDiscovery",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="number of load repetitions per file for timing (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="directory for saved figure and CSV (default: current directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the benchmark script.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to :data:`sys.argv`).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_level = getattr(logging, args.log_level, logging.WARNING)
    logging.getLogger().setLevel(log_level)

    from python_magnetrun.utils.files import expand_input_files

    datadir = {
        ".tdms": args.pigbrother_datadir,
        ".txt": args.pupitre_datadir,
        ".csv": args.pupitre_datadir,
    }
    housing = args.housing if args.housing not in ("notdefined", "") else ""
    resolved = expand_input_files(args.input_file, datadir, housing or None)

    if not resolved:
        print("No input files found.", file=sys.stderr)
        return 1

    pairs = _collect_files(
        resolved, housing, args.pigbrother_datadir, args.pupitre_datadir,
        discovery=not args.no_discovery,
    )
    if not pairs:
        print("No files to benchmark.", file=sys.stderr)
        return 1

    print(f"Benchmarking {len(pairs)} file(s) with {args.repeat} repetition(s)...")
    results = _benchmark(pairs, housing, args.repeat)

    df = pd.DataFrame(
        {
            "filepath": [r.filepath for r in results],
            "category": [r.category for r in results],
            "ext": [r.ext for r in results],
            "housing": [r.housing for r in results],
            "size_mb": [r.size_mb for r in results],
            "n_rows": [r.n_rows for r in results],
            "time_min_s": [r.time_min_s for r in results],
            "time_mean_s": [r.time_mean_s for r in results],
            "error": [r.error for r in results],
        }
    )

    _print_stats(df)

    if args.save is not None:
        csv_path = args.output_dir / "benchmark_loading.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Raw results saved: {csv_path}")

    show = args.show
    save = args.save is not None
    if show or save:
        _plot(df, args.output_dir, save=save, show=show, title=args.title)

    return 0


if __name__ == "__main__":
    sys.exit(main())
