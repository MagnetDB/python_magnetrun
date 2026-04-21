"""Plot commands: visualise MagnetRun data."""

import contextlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..hybrid import HybridRun
from ..magnetdata_base import DataType
from ..MagnetRun import MagnetRun
from ..plotting.backend import get_backend
from ..plotting.style import PlotConfig, load_plot_config
from ..utils.downsampling import DownsampleConfig

logger = logging.getLogger(__name__)


def _flatten(nested):
    """Flatten one level of nesting (replaces matplotlib.cbook.flatten)."""
    for item in nested:
        if isinstance(item, list | tuple):
            yield from item
        else:
            yield item


def _get_df_with_time(mdata, plot_args: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return (df, actual_column_names) for *plot_args*, always including 't'.

    Pandas data has 't' as an explicit key; TDMS data includes 't' automatically
    and stores channels without their group prefix.
    """
    logger.debug(f"_get_df_with_time: plot_args={plot_args}")
    if "t" in mdata.getKeys():
        df = mdata.getData(["t"] + plot_args)
        return df, list(plot_args)
    else:
        # if 't' is not in keys, we assume TDMS data where channels are stored without group prefix and 't' is implicit
        if mdata.getType() != DataType.TDMS:
            raise RuntimeError(
                "data does not contain 't' key and is not TDMS type, cannot extract time"
            )

        # get t using group of the first key in plot_args
        first_key = plot_args[0]
        tkey: str | None = None
        if "/" in first_key:
            group = first_key.split("/")[0]
            tkey = f"{group}/t"
            if hasattr(mdata, "addTdmsTime") and tkey not in mdata.getKeys():
                mdata.addTdmsTime(group=group)

        df = mdata.extractData([tkey] + plot_args)
        logger.debug(f"_get_df_with_time: df.keys()={df.keys()} -- extracted with t")

        df.rename(columns={tkey: "t"}, inplace=True)
        col_names = [k.split("/")[-1] if "/" in k else k for k in plot_args]
        logger.debug(f"_get_df_with_time: col_names={col_names}")
        return df, col_names


def _resolve_plot_config(args) -> PlotConfig:
    """Return PlotConfig from --plot-config file, or package defaults."""
    path = getattr(args, "plot_config", None)
    if path is not None:
        try:
            return load_plot_config(path)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Could not load --plot-config %s: %s — using defaults", path, exc
            )
    return PlotConfig()


def plot_bkpts(  # noqa: PLR0913
    file: str,
    channel: str,
    symbol: str,
    unit: str,
    ts: pd.DataFrame,
    smoothed: np.ndarray,
    smoothed_der1: np.ndarray,
    smoothed_der2: np.ndarray,
    quantiles_der: float,
    peaks: np.ndarray,
    ignore_peaks: list[int],
    anomalies: list[int],
    level: int,
    window: int,
    save: bool = False,
):
    """_summary_

    :param file: _description_
    :type file: str
    :param channel: _description_
    :type channel: str
    :param symbol: _description_
    :type symbol: str
    :param unit: _description_
    :type unit: str
    :param ts: _description_
    :type ts: pd.tseries
    :param smoothed: _description_
    :type smoothed: np.ndarray
    :param smoothed_der1: _description_
    :type smoothed_der1: np.ndarray
    :param smoothed_der2: _description_
    :type smoothed_der2: np.ndarray
    :param quantiles_der: _description_
    :type quantiles_der: float
    :param peaks: _description_
    :type peaks: np.ndarray
    :param ignore_peaks: _description_
    :type ignore_peaks: list[int]
    :param anomalies: _description_
    :type anomalies: list[int]
    :param save: _description_, defaults to False
    :type save: bool, optional
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(ts.to_numpy(), label=channel, color="blue", marker="o", linestyle="None")
    ax0.plot(smoothed, label="smoothed", color="red")
    ax0.legend()
    ax0.grid()
    # ax0.set_xlabel('t [s]')
    ax0.set_ylabel(f"{symbol} [{unit:~P}]")
    ax0.set_title(f"{file}: {channel}")

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(smoothed_der2, label=channel, color="red")
    ax1.legend()
    ax1.grid()
    # ax1.set_xlabel('t [s]')
    ax1.set_title(f"Savgo filter [2nd order der]: ({level}%: {quantiles_der:.3e})")

    ax2 = plt.subplot(gs[2], sharex=ax0)
    std_ts = ts.rolling(window=window).std()
    ax2.plot(std_ts.to_numpy(), label="rolling std", color="blue")
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel("t [s]")
    ax2.set_title("Rolling std")

    if peaks.shape[0]:
        ax0.plot(peaks, smoothed[peaks], "go", label="peaks")
        ax0.legend()

        ax1.plot(peaks, smoothed_der2[peaks], "go", label="peaks")
        ax1.legend()

    if ignore_peaks:
        ax0.plot(ignore_peaks, smoothed[ignore_peaks], "yo", label="ignore peaks")
        ax0.legend()

        ax1.plot(ignore_peaks, smoothed_der2[ignore_peaks], "yo", label="ignore peaks")
        ax1.legend()

    if anomalies:
        ax0.plot(anomalies, smoothed[anomalies], "ro", label="anomalies")
        ax0.legend()

        ax1.plot(anomalies, smoothed_der2[anomalies], "ro", label="anomalies")
        ax1.legend()

    if save:
        out_path = Path.cwd() / f"{Path(file).stem}-{channel}-detect_bkpts.png"
        plt.savefig(out_path, dpi=300)
    else:
        plt.show()
    plt.close()


def _parse_display_units(unit_args: list[str] | None) -> dict[str, str] | None:
    """Parse ``--unit FIELD=UNIT`` args into a display_units dict."""
    if not unit_args:
        return None
    result = {}
    for item in unit_args:
        if "=" not in item:
            logger.warning("--unit %r ignored: expected FIELD=UNIT format", item)
            continue
        field, _, unit_str = item.partition("=")
        result[field.strip()] = unit_str.strip()
    return result or None


def _default_save_path(
    input_files: list[str],
    fields: list[str],
    backend_name: str,
) -> Path:
    """Build a default output filename in CWD (never in the input-file directory)."""
    stem = Path(input_files[-1]).stem if input_files else "output"
    suffix = ".html" if "plotly" in backend_name else ".png"
    tag = "_".join(fields[:2]) if fields else "plot"
    return Path.cwd() / f"{stem}-{tag}_vs_time{suffix}"


def _handle_output(fig, args, backend, input_files, fields, backend_name, *, dpi=300):
    """Unified save-or-show logic for all plot methods."""
    save = getattr(args, "save", None)
    show = getattr(args, "show", False)
    if save is None and not show:
        show = True
    if save is not None:
        path = (
            Path(save)
            if save
            else _default_save_path(input_files, fields, backend_name)
        )
        backend.save(fig, path, dpi=dpi)
        logger.info(f"saved to {path}")
    if show:
        backend.show(fig)


def _plot_title(input_files: list[str], items) -> str:
    if not input_files:
        return ""
    if len(input_files) == 1:
        return os.path.basename(input_files[0])
    return "-".join(_flatten(items))


def _collect_hybrid(
    inputs, args, t0: list, dfs: list, all_fields: list, all_colors: list, cfg
) -> None:
    """Append hybrid data as extra columns into *dfs*/*all_fields*/*all_colors*."""
    vs_time_hybrid = getattr(args, "vs_time_hybrid", None)
    if not vs_time_hybrid or "hybrid" not in inputs:
        return
    hrun: HybridRun = inputs["hybrid"]["data"]
    time_offset = (hrun.StartTime - t0[0]).total_seconds() if t0 else 0.0
    logger.info(f"hybrid time offset: {time_offset} s")
    for key in vs_time_hybrid:
        try:
            data, time = hrun.getData(
                key, downsample=DownsampleConfig(n_out=args.hybrid_downsample)
            )
            data = np.asarray(data, dtype=float)
            time = np.asarray(time, dtype=float) + time_offset
            col_name = f"{args.fepc_system}:{args.hybrid_date}: {key}"
            dfs.append(pd.DataFrame({"t": time, col_name: data}))
            all_fields.append(col_name)
            all_colors.append(cfg.colors.overview)
        except (KeyError, ValueError, RuntimeError) as e:
            logger.error(f"key: {key} not found in hybrid data: {e}")


def _plot_vs_time_backend(
    input_files, inputs, extensions, args, cfg, backend_name: str
) -> None:
    """plot_vs_time implementation for non-matplotlib backends.

    Data is grouped by file-extension type (pupitre .txt vs pigbrother .tdms)
    so that each group is concatenated only with files of the same type.
    This preserves dense, NaN-free t-series within each group instead of
    creating a sparse interleaved mega-DataFrame.
    """
    from collections import defaultdict

    items = args.vs_time
    title = _plot_title(input_files, items)
    output_json = getattr(args, "json", False)
    same_color_per_type: bool = getattr(args, "same_color_per_type", False)
    display_units = _parse_display_units(getattr(args, "unit", None))

    # Per-extension-type accumulators (preserving insertion order via ext_order).
    ext_order: list[str] = []
    ext_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)
    ext_fields: dict[str, list[str]] = defaultdict(list)
    ext_colors: dict[str, list[str]] = defaultdict(list)
    ext_origins: dict[str, list[str]] = defaultdict(list)
    ext_units: dict[str, list[str]] = defaultdict(list)
    t0: list = []

    for i, file in enumerate(input_files):
        f_extension = os.path.splitext(file)[-1]
        plot_args: list[str] = list(items[list(extensions.keys()).index(f_extension)])
        mrun: MagnetRun = inputs[file]["data"]
        t0.append(mrun.StartTime)
        logger.info(
            f"file[{i}] {os.path.basename(file)}: StartTime={mrun.StartTime} (UTC naive)"
        )
        mdata = mrun.getMData()

        delta_t = 0.0
        if i >= 1:
            delta_t = (mrun.StartTime - t0[0]).total_seconds()
            logger.info(
                f"  align: delta_t={delta_t:.3f} s relative to file[0] {os.path.basename(input_files[0])}"
            )

        try:
            df, col_names = _get_df_with_time(mdata, plot_args)
        except (RuntimeError, KeyError) as e:
            logger.error(f"could not load data from {file}: {e}")
            continue

        df = df.copy()
        logger.info(f"  raw t range: [{df['t'].min():.3f}, {df['t'].max():.3f}] s")
        df["t"] = df["t"] + delta_t
        logger.info(f"  aligned t range: [{df['t'].min():.3f}, {df['t'].max():.3f}] s")

        from ..magnetdata_base import _make_ureg

        _ureg = _make_ureg() if display_units else None
        field_unit_strs: list[str] = []
        for raw_key, col in zip(plot_args, col_names, strict=True):
            field_short = raw_key.split("/")[-1] if "/" in raw_key else raw_key
            tgt_unit_str = (
                display_units.get(field_short) or display_units.get(raw_key)
                if display_units
                else None
            )
            try:
                _sym, src_unit = mdata.getUnitKey(raw_key)
                src_unit_str = f"{src_unit:~P}"
            except (RuntimeError, KeyError):
                src_unit_str = "?"
            if tgt_unit_str and col in df.columns and _ureg is not None:
                try:
                    factor = (
                        _ureg.Quantity(1.0, src_unit_str).to(tgt_unit_str).magnitude
                    )
                    if factor != 1.0:
                        logger.warning(
                            f"field {field_short!r}: display unit {tgt_unit_str!r} differs from data unit {src_unit_str!r} — converting (factor={factor:.6g})"
                        )
                        df[col] = df[col] * factor
                    field_unit_strs.append(tgt_unit_str)
                except (RuntimeError, ValueError, KeyError) as exc:
                    logger.warning(
                        f"field {field_short!r}: could not convert to display unit {tgt_unit_str!r}: {exc}"
                    )
                    field_unit_strs.append(src_unit_str)
            else:
                field_unit_strs.append(tgt_unit_str or src_unit_str)

        src_color = cfg.colors.get_file_color(i, f_extension, same_color_per_type)
        basename = os.path.basename(file).replace(f_extension, "")

        if len(input_files) > 1:
            rename = {old: f"{basename}: {old}" for old in col_names}
            df = df.rename(columns=rename)
            field_names = list(rename.values())
        else:
            field_names = col_names

        slice_df = df[["t"] + field_names].copy()
        slice_df.attrs.clear()  # pint cross-registry ValueError in pd.concat

        if f_extension not in ext_order:
            ext_order.append(f_extension)
        ext_dfs[f_extension].append(slice_df)
        ext_fields[f_extension].extend(field_names)
        ext_colors[f_extension].extend([src_color] * len(field_names))
        ext_origins[f_extension].extend(col_names)
        ext_units[f_extension].extend(field_unit_strs)

    # Hybrid data gets its own group so it is never mixed with other types.
    vs_time_hybrid = getattr(args, "vs_time_hybrid", None)
    if vs_time_hybrid and "hybrid" in inputs:
        hrun: HybridRun = inputs["hybrid"]["data"]
        time_offset = (hrun.StartTime - t0[0]).total_seconds() if t0 else 0.0
        logger.info(f"hybrid time offset: {time_offset} s")
        h_ext = "hybrid"
        for key in vs_time_hybrid:
            try:
                data_h, time_h = hrun.getData(
                    key, downsample=DownsampleConfig(n_out=args.hybrid_downsample)
                )
                data_h = np.asarray(data_h, dtype=float)
                time_h = np.asarray(time_h, dtype=float) + time_offset
                col_name = f"{args.fepc_system}:{args.hybrid_date}: {key}"
                hdf = pd.DataFrame({"t": time_h, col_name: data_h})
                if h_ext not in ext_order:
                    ext_order.append(h_ext)
                ext_dfs[h_ext].append(hdf)
                ext_fields[h_ext].append(col_name)
                ext_colors[h_ext].append(cfg.colors.overview)
                ext_origins[h_ext].append(key)
                ext_units[h_ext].append("?")
            except (KeyError, ValueError, RuntimeError) as e:
                logger.error(f"key: {key} not found in hybrid data: {e}")

    if not any(ext_dfs.values()):
        logger.warning("plot_vs_time: no data to plot")
        return

    # One dense DataFrame per extension type — no cross-type concatenation.
    groups: list[dict] = []
    for ext in ext_order:
        if not ext_dfs[ext]:
            continue
        c = (
            pd.concat(ext_dfs[ext], ignore_index=True)
            .sort_values("t")
            .reset_index(drop=True)
        )
        groups.append(
            {
                "combined": c,
                "fields": list(ext_fields[ext]),
                "colors": list(ext_colors[ext]),
                "origins": list(ext_origins[ext]),
                "units": list(ext_units[ext]),
            }
        )

    b = get_backend(backend_name)
    use_subplots = getattr(args, "subplots", False)
    normalize = args.normalize

    # Pre-normalise within each group using the global max per original field name.
    if normalize and len(input_files) > 1:
        for g in groups:
            c = g["combined"]
            orig_to_cols: dict[str, list[str]] = defaultdict(list)
            for renamed, orig in zip(g["fields"], g["origins"], strict=True):
                orig_to_cols[orig].append(renamed)

            norm_factors: dict[str, float] = {}
            for orig, cols in orig_to_cols.items():
                vals = np.concatenate(
                    [
                        c[col].dropna().to_numpy(dtype=float)
                        for col in cols
                        if col in c.columns
                    ]
                )
                abs_max = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
                if abs_max == 0 or not np.isfinite(abs_max):
                    abs_max = 1.0
                norm_factors[orig] = abs_max
                logger.info(f"normalize: field {orig!r} global max = {abs_max:.4g}")

            rename_map: dict[str, str] = {}
            new_fields: list[str] = []
            for renamed, orig, unit_str in zip(
                g["fields"], g["origins"], g["units"], strict=True
            ):
                abs_max = norm_factors[orig]
                c[renamed] = c[renamed] / abs_max
                new_name = f"{renamed}  (max={abs_max:.3g} [{unit_str}])"
                rename_map[renamed] = new_name
                new_fields.append(new_name)
            g["combined"] = c.rename(columns=rename_map)
            g["fields"] = new_fields
        normalize = False  # already pre-normalised

    all_fields = [f for g in groups for f in g["fields"]]

    # Plot using the backend directly — each group provides its own dense t array,
    # so no cross-type NaN interleaving occurs.
    n_series = len(all_fields)
    if use_subplots:
        fig = b.subplots(n_series, share_x=True, style=cfg.style)
        ax_idx = 0
        for g in groups:
            c = g["combined"]
            t = c["t"].to_numpy(dtype=float)
            for fname, color, orig in zip(
                g["fields"], g["colors"], g["origins"], strict=True
            ):
                if fname not in c.columns:
                    ax_idx += 1
                    continue
                y = c[fname].to_numpy(dtype=float)
                tgt = display_units.get(orig) if display_units else None
                ylabel = f"{orig} [{tgt}]" if tgt else fname
                print(
                    f"use_subplots field {orig} from {fname} with color {color} and ylabel {ylabel}",
                    flush=True,
                )
                b.add_series(
                    fig,
                    ax_idx,
                    t,
                    y,
                    label=fname,
                    normalize=normalize,
                    color=color,
                    ylabel=ylabel,
                )
                ax_idx += 1
    else:
        fig = b.subplots(1, share_x=False, style=cfg.style)
        for g in groups:
            c = g["combined"]
            t = c["t"].to_numpy(dtype=float)
            for fname, color, orig in zip(
                g["fields"], g["colors"], g["origins"], strict=True
            ):
                if fname not in c.columns:
                    continue
                y = c[fname].to_numpy(dtype=float)
                tgt = display_units.get(orig) if display_units else None
                ylabel = f"{orig} [{tgt}]" if tgt else None
                print(
                    f"overlay field {orig} from {fname} with color {color} and ylabel {ylabel}",
                    flush=True,
                )

                b.add_series(
                    fig,
                    0,
                    t,
                    y,
                    label=fname,
                    normalize=normalize,
                    color=color,
                    ylabel=ylabel,
                )

    if title and hasattr(fig, "update_layout"):
        fig.update_layout(title_text=title)
    elif title:
        fig.suptitle(title)

    b.finalize(fig)

    if output_json:
        print(b.to_json(fig))
        return

    _handle_output(
        fig, args, b, input_files, all_fields, backend_name, dpi=cfg.style.dpi
    )


def plot_vs_time(input_files, inputs, extensions, args):
    """Plot data versus time for selected keys."""
    cfg = _resolve_plot_config(args)
    backend_name = getattr(args, "backend", "matplotlib")

    use_subplots = getattr(args, "subplots", False)
    display_units = _parse_display_units(getattr(args, "unit", None))
    normalize = getattr(args, "normalize", False)
    if backend_name != "matplotlib" or use_subplots or display_units or normalize:
        _plot_vs_time_backend(input_files, inputs, extensions, args, cfg, backend_name)
        return

    # ---- matplotlib overlay path (legacy direct implementation) ----------------
    import matplotlib.pyplot as plt

    items = args.vs_time
    logger.debug(f"items={items}")
    title = _plot_title(input_files, items)

    fig, my_ax = plt.subplots(figsize=cfg.style.figsize)

    same_color_per_type: bool = getattr(args, "same_color_per_type", False)
    legends = []
    t0 = []
    symbol, unit = None, None
    seen_units: set[str] = set()
    field_units: dict[str, str] = {}       # short key → unit_str, for mixed-unit hint
    field_pint_units: dict[str, Any] = {}  # short key → pint unit, for dimensionality check
    f_extension = ""
    file = ""
    key = ""
    for i, file in enumerate(input_files):
        f_extension = os.path.splitext(file)[-1]
        plot_args = items[list(extensions.keys()).index(f_extension)]
        logger.debug(
            f"field: {file}, plot_args: {plot_args}, f_extension:{f_extension}"
        )
        mrun: MagnetRun = inputs[file]["data"]
        t0.append(mrun.StartTime)
        mdata = mrun.getMData()
        delta_t = 0.0
        if i >= 1:
            delta_t = (mrun.StartTime - t0[0]).total_seconds()
            logger.info(f"align time axis: delta_t={delta_t} s")
            mdata.shiftTime(delta_t)

        src_color = cfg.colors.get_file_color(i, f_extension, same_color_per_type)
        for key in plot_args:
            try:
                (symbol, unit) = mdata.getUnitKey(key)
                logger.debug(f"legend: file={os.path.basename(file)!r} key={key!r} symbol={symbol!r} unit={unit!r}")
                mdata.plotData(
                    x="t",
                    y=key,
                    ax=my_ax,
                    normalize=args.normalize,
                    offset=delta_t,
                    color=src_color,
                )
                unit_str = f"{unit:~P}" if unit is not None else "?"
                seen_units.add(unit_str)
                short_key = key.split("/")[-1] if "/" in key else key
                field_units.setdefault(short_key, unit_str)
                field_pint_units.setdefault(short_key, unit)
                basename = os.path.basename(file).replace(f_extension, "")
                entry = f"{basename}: {symbol} [{unit_str}]"
                logger.debug(f"legend: appending {entry!r}")
                legends.append(entry)
                if args.normalize:
                    legends[
                        -1
                    ] += f" max={float(mdata.getData([key]).max().iloc[0]):.3f} [{unit_str}]"
            except RuntimeError as e:
                logger.error(f"key: {key} not found in {file}: {e}")
                logger.info(f"available keys: {mdata.getKeys()}")
                continue
        logger.debug(f"legend: after file {os.path.basename(file)!r}: {legends}")

    # -- Hybrid data (matplotlib path) --
    vs_time_hybrid = getattr(args, "vs_time_hybrid", None)
    if vs_time_hybrid and "hybrid" in inputs:
        hrun: HybridRun = inputs["hybrid"]["data"]
        time_offset = (hrun.StartTime - t0[0]).total_seconds() if t0 else 0.0
        for key in vs_time_hybrid:
            try:
                data, time = hrun.getData(
                    key, downsample=DownsampleConfig(n_out=args.hybrid_downsample)
                )
                data = np.asarray(data, dtype=float)
                time = np.asarray(time, dtype=float)
                data_max = float(data.max())
                if args.normalize:
                    data = (data - float(data.min())) / (data_max - float(data.min()))
                my_ax.plot(time + time_offset, data, alpha=0.7, linewidth=0.5)
                label = f"{args.fepc_system}:{args.hybrid_date}: {key}"
                legends.append(label)
                if args.normalize:
                    unit_info = hrun.getMData().getUnitKey(key)
                    unit_str = f" [{unit_info[1]:~P}]" if len(unit_info) == 2 else ""
                    legends[-1] += f" max={data_max:.3f} [{unit_str}]"
                try:
                    unit_info = hrun.getMData().getUnitKey(key)
                    if len(unit_info) == 2:
                        h_unit_str = f"{unit_info[1]:~P}" if unit_info[1] is not None else "?"
                        seen_units.add(h_unit_str)
                        if symbol is None:
                            symbol, unit = unit_info
                except (KeyError, RuntimeError):
                    pass
            except (KeyError, ValueError, RuntimeError) as e:
                logger.error(f"key: {key} not found in hybrid data: {e}")

    # Re-apply grid after all df.plot() calls (pandas resets it internally)
    if cfg.style.grid:
        my_ax.grid(True, alpha=cfg.style.grid_alpha)

    if args.normalize:
        my_ax.set_ylabel("normalized")
    elif symbol is not None and unit is not None and len(seen_units) == 1:
        my_ax.set_ylabel(f"{symbol} [{unit:~P}]")
    elif len(seen_units) > 1:
        dims: set[str] = set()
        for pu in field_pint_units.values():
            if pu is not None:
                with contextlib.suppress(AttributeError, TypeError):
                    dims.add(str(pu.dimensionality))
        unit_summary = ", ".join(f"{k} [{v}]" for k, v in field_units.items())
        if len(dims) <= 1:
            target = next(iter(seen_units))
            hint = " ".join(f"--unit {k}={target}" for k in field_units)
            logger.warning(f"Mixed units on overlay ({unit_summary}) — ylabel suppressed. To align units add: {hint}")
        else:
            logger.warning(f"Incompatible dimensions on overlay ({unit_summary}) — ylabel suppressed. Consider adding --normalize to compare shapes.")
    logger.debug(f"legend: final list ({len(legends)} entries): {legends}")
    if len(legends) > 1:
        handles, auto_labels = my_ax.get_legend_handles_labels()
        logger.debug(f"legend: axes has {len(handles)} handles, auto_labels={auto_labels}")
        for handle, label in zip(handles, legends, strict=False):
            handle.set_label(label)
        leg = my_ax.legend(loc=cfg.style.legend_loc)
        logger.debug(f"legend: leg={leg}, visible={leg.get_visible() if leg is not None else 'N/A'}")
    try:
        (t_symbol, t_unit) = inputs[input_files[0]]["data"].getMData().getUnitKey("t")
        my_ax.set_xlabel(f"{t_symbol} [{t_unit:~P}]")
    except (KeyError, RuntimeError):
        my_ax.set_xlabel("t [s]")
    my_ax.set_title(title, fontsize=cfg.style.title_fontsize)
    my_ax.tick_params(labelsize=cfg.style.label_fontsize)

    b = get_backend("matplotlib")
    fields = list(_flatten(args.vs_time)) if args.vs_time else []
    _handle_output(fig, args, b, input_files, fields, "matplotlib", dpi=cfg.style.dpi)
    plt.close()


def plot_key_vs_key(input_files, inputs, extensions, args):
    """Plot key versus key pairs.

    :param input_files: List of input file paths
    :type input_files: list
    :param inputs: Dictionary containing MagnetRun data for each file
    :type inputs: dict
    :param extensions: Dictionary mapping file extensions to indices
    :type extensions: dict
    :param args: Parsed command line arguments
    :type args: argparse.Namespace
    """
    backend_name = getattr(args, "backend", "matplotlib")
    if backend_name != "matplotlib":
        logger.error(
            f"plot_key_vs_key does not yet support backend={backend_name!r}; "
            "use --backend matplotlib (or omit --backend)."
        )
        return

    import matplotlib.pyplot as plt

    cfg = _resolve_plot_config(args)
    fig, my_ax = plt.subplots(figsize=cfg.style.figsize)

    items = args.key_vs_key
    file = ""
    f_extension = ""
    title = os.path.basename(input_files[0])
    if len(input_files) > 1:
        klabels = list(_flatten(items))
        title = f"{'-'.join(klabels)}"

    legends = []
    # split pairs in key1, key2
    print(f"key_vs_key={args.key_vs_key}", flush=True)
    print(f"extensions={extensions}", flush=True)
    pairs = args.key_vs_key
    for file in input_files:
        f_extension = os.path.splitext(file)[-1]
        basename = os.path.basename(file).replace(f_extension, "")
        plot_args = pairs[list(extensions.keys()).index(f_extension)]
        logger.debug(
            f"field: {file}, plot_args: {plot_args}, f_extension:{f_extension}"
        )
        mrun: MagnetRun = inputs[file]["data"]
        mdata = mrun.getMData()

        for pair in plot_args:
            items = pair.split("-")
            if len(items) != 2:
                raise RuntimeError(f"invalid pair of keys:{pair}")
            key1 = items[0]
            key2 = items[1]
            print(f"plotting {key1} vs {key2} from {file}", flush=True)
            try:
                mdata.plotData(x=key1, y=key2, ax=my_ax)
                legends.append(f"{basename}: {key1} vs {key2}")
            except RuntimeError as e:
                logger.error(f"pair {pair!r}: key not found in {file}: {e}")
                logger.info(f"available keys: {mdata.getKeys()}")
                continue

    # Re-apply grid after all df.plot() calls (pandas resets it internally)
    if cfg.style.grid:
        my_ax.grid(True, alpha=cfg.style.grid_alpha)

    if len(legends) > 1:
        my_ax.legend(labels=legends, loc=cfg.style.legend_loc)
    my_ax.set_title(title, fontsize=cfg.style.title_fontsize)
    my_ax.tick_params(labelsize=cfg.style.label_fontsize)

    kv_fields = list(_flatten(args.key_vs_key)) if args.key_vs_key else []
    b_mpl = get_backend("matplotlib")
    _handle_output(
        fig, args, b_mpl, input_files, kv_fields, "matplotlib", dpi=cfg.style.dpi
    )
    plt.close()
