"""Plot commands: visualise MagnetRun data."""

import argparse
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..hybrid import HybridRun
from ..magnetdata_base import DataType
from ..MagnetRun import MagnetRun
from ..plotting.backend import get_backend
from ..plotting.style import PlotConfig, load_plot_config
from ..plotting.timeseries import plot_overlay, plot_subplots, plot_xy
from ..utils.downsampling import DownsampleConfig

logger = logging.getLogger(__name__)


def _flatten(nested):
    """Flatten one level of nesting (replaces matplotlib.cbook.flatten)."""
    for item in nested:
        if isinstance(item, list | tuple):
            yield from item
        else:
            yield item


# Regex for field style spec: [LINESTYLE][MARKER][:N][@ALPHA]
# LINESTYLE: '-', '--', '-.'
# MARKER:    any single or multi-char matplotlib marker string (e.g. 'o', '+', 's', 'D')
# :N:        markevery integer
# @ALPHA:    opacity float in [0, 1]
_FIELD_STYLE_RE = re.compile(r"^(-{1,2}\.?)?([^:@]+)?(?::(\d+))?(?:@([\d.]+))?$")


def parse_field_style_spec(
    spec: str,
) -> tuple[str | None, str | None, int | None, float | None]:
    """Parse a style spec string into ``(linestyle, marker, markevery, alpha)``.

    Syntax: ``[LINESTYLE][MARKER][:N][@ALPHA]``

    Examples::

        '-'         → lines only          (linestyle='-',    marker=None, markevery=None, alpha=None)
        'o'         → markers only        (linestyle='none', marker='o',  markevery=None, alpha=None)
        'o:10'      → markers every 10 pt (linestyle='none', marker='o',  markevery=10,   alpha=None)
        '-o:5'      → lines + markers/5pt (linestyle='-',    marker='o',  markevery=5,    alpha=None)
        '--s'       → dashed + square mk  (linestyle='--',   marker='s',  markevery=None, alpha=None)
        '-@0.5'     → lines, 50% opacity  (linestyle='-',    marker=None, markevery=None, alpha=0.5)
        '-o:5@0.3'  → all options         (linestyle='-',    marker='o',  markevery=5,    alpha=0.3)
    """
    m = _FIELD_STYLE_RE.match(spec)
    if not m:
        raise ValueError(f"invalid field style spec: {spec!r}")
    ls_part, mk_part, ev_part, al_part = m.group(1), m.group(2), m.group(3), m.group(4)
    # No linestyle given but marker present → suppress lines
    linestyle = ls_part if ls_part is not None else ("none" if mk_part else None)
    marker = mk_part if mk_part else None
    markevery = int(ev_part) if ev_part else None
    alpha = float(al_part) if al_part else None
    return linestyle, marker, markevery, alpha


def _parse_field_styles(
    field_style_args: list[str] | None,
) -> dict[str, tuple[str | None, str | None, int | None, float | None]]:
    """Parse a list of ``FIELD=STYLESPEC`` strings.

    The returned dict is keyed by both the full key and its short (post-``/``)
    component so look-ups work for both ``group/channel`` and bare ``channel``
    forms.
    """
    result: dict[str, tuple[str | None, str | None, int | None, float | None]] = {}
    if not field_style_args:
        return result
    for item in field_style_args:
        if "=" not in item:
            raise ValueError(f"--field_style expects FIELD=STYLESPEC, got {item!r}")
        field, spec = item.split("=", 1)
        parsed = parse_field_style_spec(spec)
        result[field] = parsed
        # also register short key so both 'group/chan' and 'chan' match
        short = field.split("/")[-1] if "/" in field else field
        result.setdefault(short, parsed)
    return result


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
            logger.warning(f"--unit {item!r} ignored: expected FIELD=UNIT format")
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


def _plot_vs_time_backend(
    input_files, inputs, extensions, args, cfg, backend_name: str
) -> None:
    """Core implementation of plot_vs_time, used for all backends.

    Data is grouped by file-extension type (pupitre .txt vs pigbrother .tdms)
    so that each group is concatenated only within its type. Groups are then
    merged into one DataFrame for plot_subplots/plot_overlay; cross-group
    columns are NaN which both matplotlib and plotly render as line gaps.
    """
    from collections import defaultdict

    from ..plotting.utils import resolve_legend_labels

    logger.info(f"plot_vs_time: input_files={input_files}, backend={backend_name}")

    items = args.vs_time
    title = getattr(args, "title", None) or _plot_title(input_files, items)
    output_json = getattr(args, "json", False)
    same_color_per_type: bool = getattr(args, "same_color_per_type", False)
    display_units = _parse_display_units(getattr(args, "display_unit", None))
    field_styles = _parse_field_styles(getattr(args, "field_style", None))

    # Pre-compute disambiguated labels from the first file's FieldMeta, identical
    # to the matplotlib path: JSON label > symbol_suffix > field name.
    _all_keys: list[str] = list(
        dict.fromkeys(k for ext_args in items for k in ext_args)
    )
    _first_mdata = inputs[input_files[0]]["data"].getMData()
    _field_metas = {k: _first_mdata.getFieldMeta(k) for k in _all_keys}
    _resolved_labels = resolve_legend_labels(_all_keys, _field_metas)

    # Per-extension-type accumulators (preserving insertion order via ext_order).
    ext_order: list[str] = []
    ext_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)
    ext_fields: dict[str, list[str]] = defaultdict(list)
    ext_colors: dict[str, list[str]] = defaultdict(list)
    ext_origins: dict[str, list[str]] = defaultdict(list)
    ext_units: dict[str, list[str]] = defaultdict(list)
    ext_styles: dict[str, list[tuple]] = defaultdict(
        list
    )  # (linestyle, marker, markevery, alpha)
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

        # Map raw col_names → resolved display labels (clash-aware) with unit suffix
        display_labels = [
            f"{_resolved_labels.get(raw_key, col)} [{unit_str}]"
            for raw_key, col, unit_str in zip(
                plot_args, col_names, field_unit_strs, strict=True
            )
        ]

        if len(input_files) > 1:
            rename = {
                old: f"{basename}: {lbl}"
                for old, lbl in zip(col_names, display_labels, strict=True)
            }
            df = df.rename(columns=rename)
            field_names = list(rename.values())
        else:
            rename = {
                old: lbl
                for old, lbl in zip(col_names, display_labels, strict=True)
                if old != lbl
            }
            if rename:
                df = df.rename(columns=rename)
            field_names = display_labels

        slice_df = df[["t"] + field_names].copy()
        slice_df.attrs.clear()  # pint cross-registry ValueError in pd.concat

        if f_extension not in ext_order:
            ext_order.append(f_extension)
        ext_dfs[f_extension].append(slice_df)
        ext_fields[f_extension].extend(field_names)
        ext_colors[f_extension].extend([src_color] * len(field_names))
        ext_origins[f_extension].extend(col_names)
        ext_units[f_extension].extend(field_unit_strs)
        for _orig_key in col_names:
            _short = _orig_key.split("/")[-1] if "/" in _orig_key else _orig_key
            _fs = field_styles.get(_orig_key) or field_styles.get(_short)
            if _fs:
                ext_styles[f_extension].append(_fs)
            else:
                ext_styles[f_extension].append((None, None, None, None))

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
                _short_h = key.split("/")[-1] if "/" in key else key
                _fs_h = field_styles.get(key) or field_styles.get(_short_h)
                ext_styles[h_ext].append(_fs_h if _fs_h else (None, None, None, None))
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
        logger.debug(
            f"ext={ext}: combined df columns: {c.columns.tolist()} -- ext_fields: {ext_fields[ext]}"
        )
        logger.debug(f"ext={ext}: combined df head:\n{c.head()}")
        groups.append(
            {
                "combined": c,
                "fields": list(ext_fields[ext]),
                "colors": list(ext_colors[ext]),
                "origins": list(ext_origins[ext]),
                "units": list(ext_units[ext]),
                "styles": list(ext_styles[ext]),
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
    all_colors = [c for g in groups for c in g["colors"]]
    all_styles = [s for g in groups for s in g["styles"]]

    # Merge all groups into one DataFrame. Each group contributes its own
    # dense t-series; columns absent in a group become NaN, which matplotlib
    # and plotly both render as line gaps — visually correct for data from
    # different sources with non-overlapping time ranges.
    merged_df = (
        pd.concat([g["combined"] for g in groups], ignore_index=True)
        .sort_values("t", kind="stable")
        .reset_index(drop=True)
    )
    logger.debug(
        f"merged_df columns: {merged_df.columns.tolist()} -- all_fields: {all_fields}"
    )
    logger.debug(f"merged_df:\n{merged_df.head()}")

    # Warn when overlaying fields whose units differ across groups.
    if not use_subplots:
        all_units_flat = [u for g in groups for u in g["units"]]
        unique_units = {u for u in all_units_flat if u != "?"}
        if len(unique_units) > 1:
            field_unit_info = ", ".join(
                f"{f!r} [{u}]"
                for g in groups
                for f, u in zip(g["fields"], g["units"], strict=True)
            )
            logger.warning(
                "Overlaying fields with different units: %s — consider using --subplots or --normalize",
                field_unit_info,
            )

    _plot_fn = plot_subplots if use_subplots else plot_overlay
    fig = _plot_fn(
        merged_df,
        all_fields,
        backend=b,
        style=cfg.style,
        title=title,
        colors=all_colors,
        field_styles=all_styles,
        normalize=normalize,
    )

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
    _plot_vs_time_backend(input_files, inputs, extensions, args, cfg, backend_name)


def plot_key_vs_key(input_files, inputs, extensions, args):
    """Plot key versus key pairs."""
    cfg = _resolve_plot_config(args)
    backend_name = getattr(args, "backend", "matplotlib")
    b = get_backend(backend_name)

    title = getattr(args, "title", None) or os.path.basename(input_files[0])
    if not getattr(args, "title", None) and len(input_files) > 1:
        klabels = list(_flatten(args.key_vs_key))
        title = "-".join(klabels)

    no_lines: bool = getattr(args, "no_lines", False)
    marker: str | None = getattr(args, "marker", None)
    if no_lines and marker is None:
        marker = "o"
    linestyle: str | None = "none" if no_lines else None

    field_styles = _parse_field_styles(getattr(args, "field_style", None))

    pairs: list[tuple[str, str]] = []
    labels: list[str] = []
    colors: list[str] = []
    dfs: list[pd.DataFrame] = []
    per_styles: list[tuple[str | None, str | None, int | None]] = []
    x_axis_labels: list[str] = []
    y_axis_labels: list[str] = []

    for i, file in enumerate(input_files):
        f_extension = os.path.splitext(file)[-1]
        basename = os.path.basename(file).replace(f_extension, "")
        plot_args = args.key_vs_key[list(extensions.keys()).index(f_extension)]
        logger.debug(
            f"field: {file}, plot_args: {plot_args}, f_extension:{f_extension}"
        )
        mrun: MagnetRun = inputs[file]["data"]
        mdata = mrun.getMData()
        color = cfg.colors.get_file_color(i, f_extension, False)

        for pair_spec in plot_args:
            parts = pair_spec.split("-")
            if len(parts) != 2:
                raise RuntimeError(f"invalid pair of keys: {pair_spec!r}")
            key1, key2 = parts
            logger.debug(f"extracting {key1} vs {key2} from {file}")
            try:
                if mdata.getType() != DataType.TDMS:
                    df_pair = mdata.getData([key1, key2])
                    col1, col2 = key1, key2
                else:
                    df_pair = mdata.extractData([key1, key2])
                    col1 = key1.split("/")[-1] if "/" in key1 else key1
                    col2 = key2.split("/")[-1] if "/" in key2 else key2
            except (RuntimeError, KeyError) as e:
                logger.error(f"pair {pair_spec!r}: key not found in {file}: {e}")
                logger.info(f"available keys: {mdata.getKeys()}")
                continue

            if len(input_files) > 1:
                new_col1, new_col2 = f"{basename}: {col1}", f"{basename}: {col2}"
                df_pair = df_pair.rename(columns={col1: new_col1, col2: new_col2})
                col1, col2 = new_col1, new_col2

            dfs.append(df_pair[[col1, col2]])
            pairs.append((col1, col2))
            lbl = (
                f"{basename}: {key1} vs {key2}"
                if len(input_files) > 1
                else f"{key1} vs {key2}"
            )
            labels.append(lbl)
            colors.append(color)
            # Look up field_style by the original y-key (key2) or its short form.
            y_short = key2.split("/")[-1] if "/" in key2 else key2
            _fs = field_styles.get(key2) or field_styles.get(y_short)
            per_styles.append(_fs if _fs else (None, None, None, None))

            # Build "symbol [unit]" axis labels from field metadata.
            for orig_key, axis_label_list in (
                (key1, x_axis_labels),
                (key2, y_axis_labels),
            ):
                try:
                    sym, unit = mdata.getUnitKey(orig_key)
                    short = orig_key.split("/")[-1] if "/" in orig_key else orig_key
                    name = sym if sym else short
                    lbl_str = f"{name} [{unit:~P}]" if unit is not None else name
                except (RuntimeError, KeyError):
                    lbl_str = orig_key.split("/")[-1] if "/" in orig_key else orig_key
                if lbl_str not in axis_label_list:
                    axis_label_list.append(lbl_str)

    if not pairs:
        logger.warning("plot_key_vs_key: no data to plot")
        return

    merged_df = pd.concat(dfs, axis=1)
    fig = plot_xy(
        merged_df,
        pairs,
        labels=labels,
        backend=b,
        style=cfg.style,
        title=title,
        colors=colors,
        marker=marker,
        linestyle=linestyle,
        field_styles=per_styles if field_styles else None,
        xlabel=", ".join(x_axis_labels) if x_axis_labels else None,
        ylabel=", ".join(y_axis_labels) if y_axis_labels else None,
    )

    b.finalize(fig)
    kv_fields = list(_flatten(args.key_vs_key)) if args.key_vs_key else []
    _handle_output(
        fig, args, b, input_files, kv_fields, backend_name, dpi=cfg.style.dpi
    )


def _run(args: "argparse.Namespace") -> int:
    import logging

    from ..log_utils import setup_logging
    from ._shared import load_inputs

    log_level = getattr(logging, getattr(args, "log_level", "WARNING").upper(), logging.WARNING)
    setup_logging(level=log_level, log_file=getattr(args, "log_file", None))

    input_files, inputs, extensions = load_inputs(args)
    if not inputs:
        logger.error("No files loaded.")
        return 1

    # optionally load hybrid data
    if getattr(args, "hybrid_datadir", None) and getattr(args, "hybrid_date", None):
        import sys
        import traceback

        from ..hybrid import HybridRun
        from ..log_utils import format_exception_location

        try:
            hrun = HybridRun.fromdir(
                base_dir=args.hybrid_datadir,
                date_str=args.hybrid_date,
                fepc_system=getattr(args, "fepc_system", "FEPC-LNCMI"),
                site=getattr(args, "site", "") or "",
            )
            inputs["hybrid"] = {"data": hrun}
        except (OSError, ValueError, RuntimeError) as e:
            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            logger.error(f"hybrid data: load error at {format_exception_location()}: {e}")
            logger.debug(f"Traceback:\n{tb_str}")

    if getattr(args, "vs_time", None):
        assert len(args.vs_time) == len(extensions), (
            f"expected {len(extensions)} --vs_time groups, got {len(args.vs_time)}"
        )
        plot_vs_time(input_files, inputs, extensions, args)

    if getattr(args, "key_vs_key", None):
        assert len(args.key_vs_key) == len(extensions), (
            f"expected {len(extensions)} --key_vs_key groups, got {len(args.key_vs_key)}"
        )
        plot_key_vs_key(input_files, inputs, extensions, args)

    return 0


def register(sub: "argparse._SubParsersAction") -> None:
    from ..cli_args import (
        create_base_parser,
        create_common_plot_parser,
        create_hybrid_parser,
        create_managed_plots_parser,
    )

    base = create_base_parser(add_input_file=False)
    plot_parser = create_common_plot_parser()
    hybrid_parser = create_hybrid_parser()
    managed_plots_parser = create_managed_plots_parser()

    p = sub.add_parser(
        "plot",
        parents=[base, plot_parser, hybrid_parser, managed_plots_parser],
        help="plot run data vs time or key vs key",
    )
    p.add_argument("input_file", nargs="+", help="input file(s)")
    p.set_defaults(_handler=_run)
