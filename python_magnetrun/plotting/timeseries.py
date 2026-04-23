"""Core time-series plotting functions.

``plot_subplots``
    N fields as stacked subplots sharing a time axis.

``plot_overlay``
    N fields on one axes, with optional per-series normalisation.

Both functions are backend-agnostic: pass ``backend="plotly"`` to get an
interactive Plotly figure instead of a matplotlib figure.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .backend import PlottingBackend, get_backend
from .style import PlotStyle

if TYPE_CHECKING:
    from ..utils.downsampling import DownsampleConfig

logger = logging.getLogger(__name__)

__all__ = ["plot_subplots", "plot_overlay", "plot_xy"]


def _resolve_backend(backend: str | PlottingBackend) -> PlottingBackend:
    if isinstance(backend, str):
        return get_backend(backend)
    return backend


def _apply_downsample(
    df: pd.DataFrame,
    t_col: str,
    fields: list[str],
    downsample: DownsampleConfig | None,
) -> pd.DataFrame:
    if downsample is None:
        return df
    try:
        from ..utils.downsampling import downsample_dataframe

        return downsample_dataframe(df, t_col, fields, downsample)
    except ImportError:
        logger.warning("utils.downsampling not available — skipping downsampling")
        return df


def _resolve_units(
    data: pd.DataFrame,
    fields: list[str],
    display_units: dict[str, str] | None,
) -> dict[str, tuple[np.ndarray, str]]:
    """Return ``{field: (values, display_symbol)}`` for each field in *fields*.

    Reads stored unit metadata from ``data.attrs["units"]``, which is expected
    to be a ``dict[str, tuple[str, pint.Unit]]`` populated by ``getData()``.
    When *display_units* names a target unit for a field **and** a stored pint
    unit is available, the values are converted via pint.  Otherwise the stored
    symbol (or field name as fallback) is used unchanged.

    .. warning::
        ``df.attrs`` is not preserved by most pandas operations (groupby, merge,
        concat, resample …).  Always read attrs immediately from the DataFrame
        passed in; never assume they survived a prior transform.
    """
    stored: dict[str, tuple[str, Any]] = data.attrs.get("units", {})
    result: dict[str, tuple[np.ndarray, str]] = {}

    for field in fields:
        if field not in data.columns:
            continue
        y = data[field].to_numpy(dtype=float)

        entry = stored.get(field)
        src_symbol: str = entry[0] if entry else field
        src_pint_unit = entry[1] if entry else None

        target_symbol = src_symbol

        if display_units and field in display_units:
            tgt_str = display_units[field]
            if src_pint_unit is not None:
                # Convert via pint using the application registry to avoid
                # cross-registry incompatibility when attrs come from getData().
                try:
                    import pint

                    ureg = pint.get_application_registry()
                    src_unit = ureg.Unit(str(src_pint_unit))
                    tgt_unit = ureg.Unit(tgt_str)
                    y = (y * src_unit).to(tgt_unit).magnitude
                    target_symbol = f"{tgt_unit:~P}"
                    if str(src_unit) != str(tgt_unit):
                        logger.warning(
                            "display unit for %r changed from %r to %r",
                            field,
                            f"{src_unit:~P}",
                            target_symbol,
                        )
                except (ImportError, ValueError) as exc:
                    logger.warning(
                        "Unit conversion for %r failed (%s) — using display_units value as symbol",
                        field,
                        exc,
                    )
                    target_symbol = tgt_str
            else:
                # No stored pint unit: use display_units value as the symbol only.
                if tgt_str != src_symbol:
                    logger.warning(
                        "display unit for %r overridden to %r (no stored pint unit — no conversion performed)",
                        field,
                        tgt_str,
                    )
                target_symbol = tgt_str
        elif src_pint_unit is not None:
            try:
                import pint

                ureg = pint.get_application_registry()
                u = ureg.Unit(str(src_pint_unit))
                target_symbol = f"{u:~P}"
            except (ImportError, ValueError):
                target_symbol = src_symbol

        result[field] = (y, target_symbol)

    return result


def _check_dimension_consistency(
    fields: list[str],
    stored: dict[str, tuple[str, Any]],
) -> None:
    """Warn when *fields* have incompatible physical dimensions."""
    dims: dict[str, str] = {}
    for f in fields:
        entry = stored.get(f)
        if entry and entry[1] is not None:
            with contextlib.suppress(AttributeError):
                dims[f] = str(entry[1].dimensionality)
    unique_dims = set(dims.values())
    if len(unique_dims) > 1:
        warnings.warn(
            f"plot_overlay: fields have incompatible dimensions {dims}. "
            "Consider normalize=True to compare shapes.",
            UserWarning,
            stacklevel=3,
        )


def plot_subplots(
    data: pd.DataFrame,
    fields: list[str],
    t_col: str = "t",
    *,
    normalize: bool = False,
    display_units: dict[str, str] | None = None,
    downsample: DownsampleConfig | None = None,
    backend: str | PlottingBackend = "matplotlib",
    style: PlotStyle | None = None,
    title: str = "",
    colors: list[str] | None = None,
    field_styles: list[tuple[str | None, str | None, int | None]] | None = None,
) -> Any:
    """Plot *N* fields as stacked subplots sharing a common time axis.

    Parameters
    ----------
    data:
        DataFrame containing *t_col* and all columns in *fields*.
    fields:
        Column names to plot (one subplot per field).
    t_col:
        Name of the time column.
    normalize:
        When ``True``, each series is divided by its absolute maximum.
        The legend label is amended: ``"field  (max=X.XX)"``.
    display_units:
        Optional mapping ``{field: target_unit}`` (pint-parseable strings).
        Stored units are read from ``data.attrs["units"]`` and converted before
        plotting.  When no stored pint unit is available, the value is used as
        a plain symbol override.  The resolved symbol appears in each subplot's
        ylabel as ``"field [symbol]"``.
    downsample:
        Optional ``DownsampleConfig``.  Raises ``ValueError`` for
        ``"plotly-resampler"`` / ``"plotly-widget"`` backends.
    backend:
        Backend name string or a pre-constructed ``PlottingBackend`` instance.
    style:
        ``PlotStyle`` configuration.
    title:
        Optional figure title (set on the first subplot / suptitle).
    colors:
        Optional per-series colour list.
    field_styles:
        Optional per-series style list of ``(linestyle, marker, markevery)``
        tuples.  Use ``None`` inside a tuple to leave that attribute at the
        backend default.  Example::

            field_styles=[("-", "o", 10), ("--", None, None)]

    Returns
    -------
    Any
        Opaque figure handle.  Call ``backend.save(fig, path)`` or
        ``backend.show(fig)`` to output it.
    """
    if not fields:
        raise ValueError("fields must not be empty")

    b = _resolve_backend(backend)

    _backend_name = backend if isinstance(backend, str) else type(backend).__name__
    live_kernel = _backend_name in ("plotly-resampler", "plotly-widget")
    if live_kernel and downsample is not None:
        raise ValueError(
            f"backend={_backend_name!r} performs dynamic view-dependent resampling "
            "(tier 3) and must receive the full-resolution data. "
            "Do not pass a DownsampleConfig — remove the downsample argument."
        )
    df = _apply_downsample(data, t_col, fields, downsample)

    t = df[t_col].to_numpy(dtype=float)

    resolved = _resolve_units(df, fields, display_units)

    fig = b.subplots(len(fields), share_x=True, style=style)

    for i, field in enumerate(fields):
        if field not in df.columns:
            logger.warning("Field %r not found in DataFrame — skipping", field)
            continue
        y, symbol = resolved.get(field, (df[field].to_numpy(dtype=float), field))
        color = colors[i] if colors and i < len(colors) else None
        ls, mk, me = field_styles[i] if field_styles and i < len(field_styles) else (None, None, None)
        ylabel = f"{field} [{symbol}]" if symbol != field else field
        b.add_series(
            fig, i, t, y, label=field, normalize=normalize,
            color=color, ylabel=ylabel,
            linestyle=ls, marker=mk, markevery=me,
        )

    if title and hasattr(fig, "update_layout"):
        fig.update_layout(title_text=title)
    elif title:
        fig.suptitle(title)

    return fig


def plot_overlay(
    data: pd.DataFrame,
    fields: list[str],
    t_col: str = "t",
    *,
    normalize: bool = False,
    display_units: dict[str, str] | None = None,
    downsample: DownsampleConfig | None = None,
    backend: str | PlottingBackend = "matplotlib",
    style: PlotStyle | None = None,
    title: str = "",
    colors: list[str] | None = None,
    field_styles: list[tuple[str | None, str | None, int | None]] | None = None,
) -> Any:
    """Plot *N* fields on a single axes (overlay).

    Parameters
    ----------
    data:
        DataFrame containing *t_col* and all columns in *fields*.
    fields:
        Column names to overlay on the same axes.
    t_col:
        Name of the time column.
    normalize:
        When ``True``, each series is divided by its absolute maximum before
        plotting.  Legend label format:

        - same unit for all fields → ``"field  (max=X.XX)"``
        - mixed units             → ``"field [unit]  (max=X.XX)"``
    display_units:
        Optional mapping ``{field: target_unit}`` (pint-parseable strings).
        Stored units are read from ``data.attrs["units"]`` and converted.
        When no stored pint unit is available, the value is used as a plain
        symbol override.

        **ylabel rules:**

        - all curves resolve to the **same** display symbol → shared ylabel
          ``"[symbol]"``
        - symbols differ → no ylabel; each legend entry carries the unit:
          ``"field [symbol]"``

        A dimension-consistency warning is emitted when fields from
        ``data.attrs["units"]`` have incompatible physical dimensions;
        ``normalize=True`` is suggested as a workaround.
    downsample:
        Optional ``DownsampleConfig``.  Raises ``ValueError`` for
        ``"plotly-resampler"`` / ``"plotly-widget"`` backends.
    backend:
        Backend name or instance.
    style:
        ``PlotStyle`` configuration.
    title:
        Optional figure title.
    colors:
        Optional per-series colour list.
    field_styles:
        Optional per-series style list of ``(linestyle, marker, markevery)``
        tuples.  Use ``None`` inside a tuple to leave that attribute at the
        backend default.

    Returns
    -------
    Any
        Opaque figure handle.
    """
    if not fields:
        raise ValueError("fields must not be empty")

    b = _resolve_backend(backend)

    _backend_name = backend if isinstance(backend, str) else type(backend).__name__
    live_kernel = _backend_name in ("plotly-resampler", "plotly-widget")
    if live_kernel and downsample is not None:
        raise ValueError(
            f"backend={_backend_name!r} performs dynamic view-dependent resampling "
            "(tier 3) and must receive the full-resolution data. "
            "Do not pass a DownsampleConfig — remove the downsample argument."
        )
    df = _apply_downsample(data, t_col, fields, downsample)

    # Warn on dimension mismatches before doing anything else.
    stored: dict = df.attrs.get("units", {})
    _check_dimension_consistency(fields, stored)

    t = df[t_col].to_numpy(dtype=float)

    resolved = _resolve_units(df, fields, display_units)

    # Determine ylabel strategy from the resolved symbols.
    present_fields = [f for f in fields if f in df.columns]
    symbols = [resolved[f][1] for f in present_fields if f in resolved]
    unique_symbols = set(symbols)
    mixed_units = len(unique_symbols) > 1
    shared_symbol = next(iter(unique_symbols)) if len(unique_symbols) == 1 else None
    # A symbol equal to the field name means no unit metadata was available.
    has_unit_info = shared_symbol is not None and shared_symbol not in present_fields
    shared_ylabel = f"[{shared_symbol}]" if (not mixed_units and has_unit_info) else None

    fig = b.subplots(1, share_x=False, style=style)

    for i, field in enumerate(fields):
        if field not in df.columns:
            logger.warning("Field %r not found in DataFrame — skipping", field)
            continue
        y, symbol = resolved.get(field, (df[field].to_numpy(dtype=float), field))
        color = colors[i] if colors and i < len(colors) else None
        ls, mk, me = field_styles[i] if field_styles and i < len(field_styles) else (None, None, None)

        # Build legend label according to unit/normalize rules.
        unit_in_legend = mixed_units and symbol != field
        if normalize:
            abs_max = float(np.nanmax(np.abs(y))) if len(y) else 1.0
            if abs_max == 0 or not np.isfinite(abs_max):
                abs_max = 1.0
            y = y / abs_max
            if unit_in_legend:
                label = f"{field} [{symbol}]  (max={abs_max:.3g})"
            else:
                label = f"{field}  (max={abs_max:.3g})"
            # Values already normalised here; tell backend not to re-normalise.
            do_normalize = False
        else:
            label = f"{field} [{symbol}]" if unit_in_legend else field
            do_normalize = False

        # Pass ylabel only for the first series (sets it once on the axes).
        ylabel = shared_ylabel if i == 0 else None
        b.add_series(
            fig, 0, t, y, label=label, normalize=do_normalize,
            color=color, ylabel=ylabel,
            linestyle=ls, marker=mk, markevery=me,
        )

    if title and hasattr(fig, "update_layout"):
        fig.update_layout(title_text=title)
    elif title:
        fig.suptitle(title)

    return fig


def plot_xy(
    data: pd.DataFrame,
    pairs: list[tuple[str, str]],
    *,
    labels: list[str] | None = None,
    backend: str | PlottingBackend = "matplotlib",
    style: PlotStyle | None = None,
    title: str = "",
    colors: list[str] | None = None,
    marker: str | None = None,
    linestyle: str | None = None,
    field_styles: list[tuple[str | None, str | None, int | None]] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Any:
    """Plot arbitrary (x, y) column pairs on a single axes.

    Unlike :func:`plot_subplots` / :func:`plot_overlay`, the x-axis is not
    assumed to be time.  Each entry in *pairs* names ``(x_col, y_col)``
    columns from *data*; all pairs share the same axes.  NaN values in
    either column of a pair are dropped before plotting.

    Parameters
    ----------
    data:
        DataFrame containing all x and y columns referenced in *pairs*.
    pairs:
        List of ``(x_col, y_col)`` tuples plotted as separate series.
    labels:
        Per-series legend labels.  Defaults to ``"y_col vs x_col"``.
    backend:
        Backend name or instance.
    style:
        ``PlotStyle`` configuration.
    title:
        Optional figure title.
    colors:
        Optional per-series colour list.
    marker:
        Marker style applied to all series (overridden per-series by *field_styles*).
    linestyle:
        Line style applied to all series (overridden per-series by *field_styles*).
    field_styles:
        Optional per-series ``(linestyle, marker, markevery)`` tuples, in the
        same order as *pairs*.  ``None`` entries fall back to the global
        *marker* / *linestyle* values.
    xlabel:
        Explicit x-axis label (e.g. ``"I [A]"``)  Defaults to the unique
        x-column names joined by ", ".
    ylabel:
        Explicit y-axis label (e.g. ``"U [V]"``)  Defaults to the first
        y-column name.

    Returns
    -------
    Any
        Opaque figure handle.  Call ``backend.save(fig, path)`` or
        ``backend.show(fig)`` to output it.
    """
    if not pairs:
        raise ValueError("pairs must not be empty")

    b = _resolve_backend(backend)
    fig = b.subplots(1, share_x=False, style=style)

    # Determine x-axis label from the set of unique x columns (fallback).
    x_cols_seen = list(dict.fromkeys(xc for xc, _ in pairs))
    _xlabel = xlabel if xlabel is not None else (", ".join(x_cols_seen) if x_cols_seen else "")

    for i, (x_col, y_col) in enumerate(pairs):
        if x_col not in data.columns:
            logger.warning("Column %r not found in DataFrame — skipping pair", x_col)
            continue
        if y_col not in data.columns:
            logger.warning("Column %r not found in DataFrame — skipping pair", y_col)
            continue

        # Drop rows where either coordinate is NaN so scatter/line is clean.
        pair_df = data[[x_col, y_col]].dropna()
        x = pair_df[x_col].to_numpy(dtype=float)
        y = pair_df[y_col].to_numpy(dtype=float)

        label = labels[i] if labels and i < len(labels) else f"{y_col} vs {x_col}"
        color = colors[i] if colors and i < len(colors) else None
        # Use explicit ylabel if provided, otherwise fall back to the y-column name.
        _ylabel_series = (ylabel if ylabel is not None else y_col) if i == 0 else None

        # Per-series style overrides the global marker/linestyle.
        _fs = field_styles[i] if field_styles and i < len(field_styles) else (None, None, None)
        _ls = _fs[0] if _fs[0] is not None else linestyle
        _mk = _fs[1] if _fs[1] is not None else marker
        _me = _fs[2]

        b.add_series(
            fig, 0, x, y,
            label=label,
            normalize=False,
            color=color,
            ylabel=_ylabel_series,
            marker=_mk,
            linestyle=_ls,
            markevery=_me,
        )

    # Store xlabel so finalize() uses it instead of the default "t [s]".
    fig._magnetrun_xlabel = _xlabel  # type: ignore[attr-defined]

    if title and hasattr(fig, "update_layout"):
        fig.update_layout(title_text=title)
    elif title:
        fig.suptitle(title)

    return fig
