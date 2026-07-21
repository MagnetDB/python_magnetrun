#!/usr/bin/env python3
"""Example: inspect field metadata, groups, and downsampling quality.

Loads a single data file (`.txt` or `.tdms`) via :func:`~python_magnetrun.MagnetRun.load_mrun`,
then prints for every field its symbol, unit, label and description, followed by
the list of fields belonging to each group, the first rows of one chosen data
column, and finally quality metrics comparing raw vs downsampled data.

Usage
-----
::

    # pupitre (.txt) — default stride downsampling
    python field_meta_example.py data/2025.11.05\\ -\\ 09:53:00.txt --housing M8

    # pigbrother (.tdms) with explicit key and LTTB method
    python field_meta_example.py data/M8_Overview_251105-0949.tdms --housing M8 \\
        --key Courants_Alimentations/Champ_magn --method lttb --n-out 500

    # minmax with custom bucket size
    python field_meta_example.py data/2025.11.05\\ -\\ 09:53:00.txt --housing M8 \\
        --method minmax --n-out 200 --bucket-size 10

    # rdp with explicit epsilon
    python field_meta_example.py data/2025.11.05\\ -\\ 09:53:00.txt --housing M8 \\
        --method rdp --n-out 300 --epsilon 0.01
"""

from __future__ import annotations

import argparse
import sys

from python_magnetrun.log_utils import get_logger, setup_logging
from python_magnetrun.MagnetRun import load_mrun

logger = get_logger(__name__)


def print_field_meta(mdata) -> None:
    """Print symbol, unit, label, and description for every field in *mdata*.

    Parameters
    ----------
    mdata : MagnetDataBase
        Loaded data object after :meth:`~python_magnetrun.magnetdata_base.MagnetDataBase.Units`
        has been called.
    """
    print(f"\n{'Field':<45} {'Symbol':<10} {'Unit':<20} {'Label':<20} Description")
    print("-" * 120)
    for key, meta in sorted(mdata.field_meta.items()):
        unit_str = f"{meta.unit:~P}" if meta.unit is not None else "—"
        print(
            f"{key:<45} {meta.symbol:<10} {unit_str:<20} {meta.label:<20} {meta.description}"
        )


def print_groups(mdata) -> None:
    """Print the fields belonging to each group defined in *mdata*.

    Parameters
    ----------
    mdata : MagnetDataBase
        Loaded data object after :meth:`~python_magnetrun.magnetdata_base.MagnetDataBase.Units`
        has been called.
    """
    groups = mdata.list_groups()
    if not groups:
        print("\n(no groups defined)")
        return

    print(f"\n{'Group':<35} Fields")
    print("-" * 80)
    for group in groups:
        raw = mdata.Groups[group]
        # pupitre: list of column names; TDMS: dict of {channel: props}
        if isinstance(raw, dict):
            channels = [ch for ch in raw if ch not in ("t", "timestamp")]
            fields = [f"{group}/{ch}" for ch in channels]
        else:
            fields = list(raw)
        print(f"{group:<35} {', '.join(fields)}")


_SKIP_KEYS = {"t", "timestamp", "Date", "Time"}


def _default_key(mdata) -> str:
    """Return the first meaningful key from *mdata.field_meta*."""
    for key in mdata.field_meta:
        if key not in _SKIP_KEYS:
            return key
    raise RuntimeError("No displayable key found in field_meta.")


_ALL_METHODS = ["stride", "lttb", "minmax_lttb", "m4", "nan_m4", "minmax", "rdp", "vw"]


def print_key_preview(mdata, key: str, n: int = 5) -> None:
    """Print metadata and the first *n* rows of the data column for *key*.

    Parameters
    ----------
    mdata : MagnetDataBase
        Loaded data object after :meth:`~python_magnetrun.magnetdata_base.MagnetDataBase.Units`
        has been called.
    key : str
        Column or ``"Group/Channel"`` key to preview.
    n : int
        Number of rows to display (default ``5``).
    """
    meta = mdata.field_meta.get(key)
    print(f"\n=== Data preview: {key!r} ===")
    if meta is not None:
        unit_str = f"{meta.unit:~P}" if meta.unit is not None else "—"
        print(f"  symbol={meta.symbol!r}  unit={unit_str}  label={meta.label!r}")
        print(f"  description: {meta.description}")
    df = mdata.getData(key)
    print(df.head(n).to_string())


def print_downsampling_metrics(
    mdata,
    key: str,
    method: str,
    n_out: int | None,
    epsilon: float | None,
    bucket_size: int | None,
) -> None:
    """Downsample the *key* column and report quality metrics against the raw signal.

    Parameters
    ----------
    mdata : MagnetDataBase
        Loaded data object.
    key : str
        Column or ``"Group/Channel"`` key to downsample.
    method : str
        Downsampling algorithm name (see :class:`~python_magnetrun.utils.downsampling.DownsampleConfig`).
    n_out : int or None
        Target number of output points; defaults to 10 % of signal length.
    epsilon : float or None
        Geometry tolerance for ``rdp``/``vw``; when ``None`` with those methods,
        auto-searched via :meth:`~python_magnetrun.utils.downsampling.DownsampleConfig.from_n_out_rdp`.
    bucket_size : int or None
        Bucket size for the ``minmax`` method; ``None`` lets the algorithm
        auto-compute it from *n_out*.
    """
    import numpy as np

    from python_magnetrun.utils.downsampling import DownsampleConfig
    from python_magnetrun.utils.downsampling_metrics import evaluate_downsampling

    df = mdata.getData(key)
    # value column: last column of the returned DataFrame
    val_col = df.columns[-1]
    data = df[val_col].to_numpy(dtype=float)
    time = np.arange(len(data), dtype=float)

    effective_n_out = n_out if n_out is not None else max(2, len(data) // 10)

    if method in ("rdp", "vw"):
        if epsilon is not None:
            config = DownsampleConfig(n_out=effective_n_out, method=method, epsilon=epsilon)
        else:
            config = DownsampleConfig.from_n_out_rdp(data, time, effective_n_out, method=method)
    elif method == "minmax":
        config = DownsampleConfig(n_out=effective_n_out, method=method, bucket_size=bucket_size)
    else:
        config = DownsampleConfig(n_out=effective_n_out, method=method)

    metrics = evaluate_downsampling(data, time, config)

    print(f"\n=== Downsampling metrics: {key!r} ===")
    print(f"  {metrics.summary()}")
    print()
    rows = [
        ("n_original",        f"{metrics.n_original}"),
        ("n_downsampled",     f"{metrics.n_downsampled}"),
        ("compression_ratio", f"{metrics.compression_ratio:.2f}x"),
        ("RMSE",              f"{metrics.rmse:.6g}"),
        ("MAE",               f"{metrics.mae:.6g}"),
        ("max_error",         f"{metrics.max_error:.6g}"),
        ("MAPE",              f"{metrics.mape:.4g} %"),
        ("hausdorff_distance",f"{metrics.hausdorff_distance:.6g}"),
        ("peak_max_error",    f"{metrics.peak_max_error:.4g}"),
        ("peak_min_error",    f"{metrics.peak_min_error:.4g}"),
        ("energy_ratio",      f"{metrics.energy_ratio:.6f}"),
        ("elapsed",           f"{metrics.elapsed_s * 1000:.2f} ms"),
    ]
    col_w = max(len(r[0]) for r in rows) + 2
    for name, value in rows:
        print(f"  {name:<{col_w}} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", metavar="FILE", help="pupitre .txt or pigbrother .tdms file")
    parser.add_argument(
        "--housing", default="unknown", help="housing name, e.g. M8 (default: unknown)"
    )
    parser.add_argument(
        "--key", default=None, help="field key to preview (default: first meaningful key)"
    )

    # --- downsampling arguments -------------------------------------------
    ds_group = parser.add_argument_group("downsampling")
    ds_group.add_argument(
        "--method",
        default="stride",
        choices=_ALL_METHODS,
        help="downsampling algorithm (default: stride)",
    )
    ds_group.add_argument(
        "--n-out",
        type=int,
        default=None,
        metavar="N",
        help="target number of output points (default: 10%% of signal length)",
    )
    ds_group.add_argument(
        "--epsilon",
        type=float,
        default=None,
        metavar="E",
        help="geometry tolerance for rdp/vw (auto-searched when omitted)",
    )
    ds_group.add_argument(
        "--bucket-size",
        type=int,
        default=None,
        metavar="N",
        help="bucket size for minmax method (auto-computed when omitted)",
    )

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.debug else "INFO")

    mrun = load_mrun(args.file, housing=args.housing)
    mdata = mrun.getMData()

    print(f"File    : {mdata.FileName}")
    print(f"Type    : {mdata.Type.name}")
    print(f"Keys    : {len(mdata.Keys)} fields")

    mdata.Units(debug=args.debug)

    print(f"\n=== Field metadata ({len(mdata.field_meta)} entries) ===")
    print_field_meta(mdata)

    print(f"\n=== Groups ({len(mdata.list_groups())} defined) ===")
    print_groups(mdata)

    key = args.key if args.key is not None else _default_key(mdata)
    print_key_preview(mdata, key)

    print_downsampling_metrics(
        mdata,
        key,
        method=args.method,
        n_out=args.n_out,
        epsilon=args.epsilon,
        bucket_size=args.bucket_size,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
