"""
field_comparison — Compare pupitre and pigbrother (Overview/Archive) timeseries.

Uses the cross-format aliases declared in ``pupitre-defs.json`` (see
:mod:`python_magnetrun.field_defs`) to discover every pupitre field that has
a pigbrother counterpart, synchronizes pupitre against each pigbrother
source using a single reference lag per ``(record, source)`` pair, and
reports similarity metrics (Euclidean, MAPE, Pearson correlation, optional
DTW) for each field.

The reference lag is computed once per source from ``Idcct1``/``Courant_A1``
(falling back to ``Idcct3``/``Courant_A3`` when the first pair isn't
available) rather than per field — the lag is a clock-synchronization
property of the whole acquisition, not of an individual channel, and the
main current channels give the strongest cross-correlation signal. If
neither pair is available, that indicates a problem in the data files, so
:func:`compute_reference_lag` raises rather than silently skipping.

Example usage::

    from python_magnetrun.analysis.processing import (
        process_overview_file,
        ProcessingConfig,
    )
    from python_magnetrun.analysis.field_comparison import (
        compare_all_fields,
        print_comparison_summary,
    )

    record = process_overview_file("M9_Overview_241106-0935.tdms", ProcessingConfig())
    results = compare_all_fields(record)
    print_comparison_summary(results)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import pint

from .. import field_defs
from ..plotting.utils import format_axis_label
from .loaders import load_files_data
from .metrics import compare_series
from .plotting import plot_comparison
from .synchronization import (
    LagResult,
    apply_lag_correction,
    compute_lag,
    compute_lag_interpolated,
)

if TYPE_CHECKING:
    from .processing import OverviewRecord

logger = logging.getLogger("python_magnetrun.analysis.field_comparison")

#: Pigbrother sources this module compares against (incidents/hybrid excluded).
PIGBROTHER_SOURCES: tuple[str, ...] = ("overview", "archive")

#: Pupitre keys tried, in order, as the reference-lag channel pair.
REFERENCE_LAG_KEYS: tuple[str, ...] = ("Idcct1", "Idcct3")

#: Point-count guard above which DTW is skipped (mirrors analysis/cli.py).
_DTW_MAX_POINTS = 5000

#: Bare coil-tap channel, e.g. ``"Interne2"``.
_TAP_CHANNEL_RE = re.compile(r"^Interne(\d+)$")

#: Merged coil-tap channel reported when two adjacent taps share one
#: reading, e.g. ``"Interne1-2"`` (tap 1 not independently wired).
_COMBINED_TAP_CHANNEL_RE = re.compile(r"^Interne(\d+)-(\d+)$")


@dataclass(frozen=True)
class AliasedField:
    """A pupitre field with a known pigbrother counterpart.

    Attributes
    ----------
    pupitre_key : str
        Column name in pupitre data.
    pigbrother_group : str
        TDMS group name in pigbrother data.
    pigbrother_channel : str
        TDMS channel name (bare, without group prefix) in pigbrother data.
    pupitre_unit : str, optional
        Unit declared for *pupitre_key* in ``pupitre-defs.json``, e.g. ``"tesla"``.
    pigbrother_unit : str, optional
        Unit declared for *pigbrother_group*/*pigbrother_channel* in
        ``pigbrother-defs.json``, e.g. ``"millitesla"``.
    pupitre_symbol : str, optional
        Symbol declared for *pupitre_key* in ``pupitre-defs.json``, e.g. ``"Bz"``.
    """

    pupitre_key: str
    pigbrother_group: str
    pigbrother_channel: str
    pupitre_unit: str | None = None
    pigbrother_unit: str | None = None
    pupitre_symbol: str | None = None


@dataclass
class FieldComparisonResult:
    """Result of comparing one field's timeseries for one pigbrother source.

    Attributes
    ----------
    field : AliasedField
        The compared field.
    source : str
        Pigbrother source: ``"overview"`` or ``"archive"``.
    available : bool
        Whether the comparison could be computed.
    reason : str, optional
        Why *available* is False (e.g. missing channel).
    n_points : int
        Number of points compared (post-interpolation, overlap window only).
    metrics : dict, optional
        Output of :func:`~python_magnetrun.analysis.metrics.compare_series`.
    plot_path : str, optional
        Path of the saved comparison plot, if requested.
    reference_lag : LagResult, optional
        The single reference lag computed for *(record, source)* and applied
        to pupitre data before this field was compared (see
        :func:`compute_reference_lag`).
    reference_field : AliasedField, optional
        The pupitre/pigbrother field pair used to estimate *reference_lag*
        (e.g. ``Idcct1``/``Courant_A1``).
    """

    field: AliasedField
    source: str
    available: bool
    reason: str | None = None
    n_points: int = 0
    metrics: dict[str, Any] | None = None
    plot_path: str | None = None
    reference_lag: LagResult | None = None
    reference_field: AliasedField | None = None


def discover_pupitre_pigbrother_fields(
    pupitre_defs_file: str = "pupitre-defs.json",
    pigbrother_defs_file: str = "pigbrother-defs.json",
) -> list[AliasedField]:
    """Discover every pupitre field with a pigbrother cross-format alias.

    Parameters
    ----------
    pupitre_defs_file : str, optional
        ``*-defs.json`` filename or path, resolved via
        :func:`~python_magnetrun.field_defs.resolve_defs_file`.
    pigbrother_defs_file : str, optional
        ``*-defs.json`` filename or path used to look up each field's
        declared pigbrother-side unit, resolved the same way.

    Returns
    -------
    list of AliasedField
        One entry per pupitre field that declares a ``"pigbrother"`` alias,
        with ``pupitre_unit``/``pigbrother_unit``/``pupitre_symbol`` populated
        from each side's defs file when declared.
    """
    defs = field_defs.load_defs(pupitre_defs_file)
    pigbrother_defs = field_defs.load_defs(pigbrother_defs_file)
    fields: list[AliasedField] = []
    for key, entry in defs.items():
        if key.startswith("_"):
            continue
        target = entry.get("aliases", {}).get("pigbrother")
        if not target:
            continue
        group, _, channel = target.partition("/")
        if not channel:
            logger.warning(
                f"discover_pupitre_pigbrother_fields: malformed pigbrother alias "
                f"for {key!r}: {target!r} (expected 'Group/Channel')"
            )
            continue
        fields.append(
            AliasedField(
                pupitre_key=key,
                pigbrother_group=group,
                pigbrother_channel=channel,
                pupitre_unit=entry.get("unit"),
                pigbrother_unit=pigbrother_defs.get(target, {}).get("unit"),
                pupitre_symbol=entry.get("symbol"),
            )
        )
    return fields


def _load_pigbrother_group(
    record: OverviewRecord,
    source: str,
    group: str,
    cache: dict[tuple[str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Load (and cache) one full pigbrother TDMS group for *source*."""
    cache_key = (source, group)
    if cache_key in cache:
        return cache[cache_key]

    if record.sources is None:
        files: list[str] = []
    elif source == "overview":
        files = record.sources.overview
    elif source == "archive":
        files = record.sources.archive
    else:
        raise ValueError(f"_load_pigbrother_group: unknown source {source!r}")

    df = (
        load_files_data(files, record.housing, group, keys=None)
        if files
        else pd.DataFrame()
    )
    cache[cache_key] = df
    return df


def _load_overview_channel_map(record: OverviewRecord) -> dict[str, set[str]]:
    """Return the real TDMS group/channel structure of *record*'s overview file(s).

    Metadata-only read via :class:`nptdms.TdmsFile` (no data arrays loaded),
    so this is cheap even for large files.

    Parameters
    ----------
    record : OverviewRecord
        Record whose ``sources.overview`` file(s) are inspected.

    Returns
    -------
    dict
        ``{group_name: {channel_name, ...}}`` merged across every file in
        ``record.sources.overview``. Empty if *record* has no sources, no
        overview files, or none of them could be opened — callers should
        treat an empty result as "unknown" and skip filtering rather than
        as "no channels exist".
    """
    from nptdms import TdmsFile

    channel_map: dict[str, set[str]] = {}
    if record.sources is None:
        return channel_map

    for file in record.sources.overview:
        try:
            with TdmsFile.open(file) as tdms:
                for group in tdms.groups():
                    channel_map.setdefault(group.name, set()).update(
                        ch.name for ch in group.channels()
                    )
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"_load_overview_channel_map: failed to read {file}: {e}")

    return channel_map


def _find_combined_tap_channel(channel: str, available: set[str]) -> str | None:
    """Find a merged-tap channel covering *channel*'s tap number, if any.

    Some housings report two adjacent coil taps as a single merged channel
    (e.g. ``"Interne1-2"``) instead of separate ``"Interne1"``/``"Interne2"``
    channels, when the low tap isn't independently wired. The merged
    channel is reported under the high tap's number, so this only matches
    when *channel* is the bare high-tap name.

    Parameters
    ----------
    channel : str
        Bare pigbrother channel name being looked up, e.g. ``"Interne2"``.
    available : set of str
        Channel names actually present in the TDMS group.

    Returns
    -------
    str or None
        Name of the merged channel covering *channel*'s tap, or ``None``
        if *channel* isn't a bare tap channel or no merged channel matches.
    """
    m = _TAP_CHANNEL_RE.match(channel)
    if m is None:
        return None
    tap = m.group(1)
    for candidate in available:
        cm = _COMBINED_TAP_CHANNEL_RE.match(candidate)
        if cm is not None and cm.group(2) == tap:
            return candidate
    return None


def _load_pupitre_fields(record: OverviewRecord, keys: list[str]) -> pd.DataFrame:
    """Load pupitre data for *keys* (plus t/timestamp), bypassing housing_config.

    Aliases are housing-independent by design, so unlike
    :func:`~python_magnetrun.analysis.processing.load_pupitre_data`, this
    does not layer in any housing-config-derived keys.
    """
    if record.sources is None or not record.sources.pupitre:
        return pd.DataFrame()
    return load_files_data(record.sources.pupitre, record.housing, group="", keys=keys)


def compute_reference_lag(
    record: OverviewRecord,
    source: str,
    pupitre_df: pd.DataFrame,
    pigbrother_group_df: pd.DataFrame,
    reference_fields: dict[str, AliasedField],
    lag_method: Literal["resample_1s", "interpolated"] = "resample_1s",
) -> tuple[LagResult, AliasedField]:
    """Compute a single reference lag for *(record, source)*.

    Tries ``Idcct1``/``Courant_A1`` first, then ``Idcct3``/``Courant_A3``.
    Both pairs missing (on either side) indicates a problem in the data
    files, not a normal fallback case, so this raises rather than returning
    a sentinel.

    Parameters
    ----------
    record : OverviewRecord
        Record being processed (used only for the error message).
    source : str
        Pigbrother source name, for the error message.
    pupitre_df : pandas.DataFrame
        Loaded pupitre data (must contain ``timestamp``).
    pigbrother_group_df : pandas.DataFrame
        Loaded pigbrother ``Courants_Alimentations`` data (must contain
        ``timestamp``).
    reference_fields : dict
        ``{"Idcct1": AliasedField(...), "Idcct3": AliasedField(...)}`` as
        returned by :func:`discover_pupitre_pigbrother_fields`.
    lag_method : {"resample_1s", "interpolated"}, optional
        Lag algorithm to use: ``"resample_1s"`` delegates to the existing
        :func:`~python_magnetrun.analysis.synchronization.compute_lag`
        (correlation/confidence unavailable, left at 0.0); ``"interpolated"``
        delegates to
        :func:`~python_magnetrun.analysis.synchronization.compute_lag_interpolated`.

    Returns
    -------
    tuple of (LagResult, AliasedField)
        The computed reference lag, and the pupitre/pigbrother field pair
        used to compute it (e.g. ``Idcct1``/``Courant_A1``).

    Raises
    ------
    ValueError
        If neither reference pair's columns are present in the loaded data.
    """
    tried = []
    for ref_key in REFERENCE_LAG_KEYS:
        af = reference_fields.get(ref_key)
        if af is None:
            continue
        tried.append(f"{af.pupitre_key}/{af.pigbrother_channel}")
        if af.pupitre_key not in pupitre_df.columns:
            continue
        if af.pigbrother_channel not in pigbrother_group_df.columns:
            continue

        df1_data = {
            "df": pigbrother_group_df[["timestamp", af.pigbrother_channel]],
            "field": af.pigbrother_channel,
            "range": {"start": 0, "end": None},
        }
        df2_data = {
            "df": pupitre_df[["timestamp", af.pupitre_key]],
            "field": af.pupitre_key,
            "range": {"start": 0, "end": None},
        }

        if lag_method == "interpolated":
            return compute_lag_interpolated("timestamp", df1_data, df2_data), af
        lag = compute_lag("timestamp", df1_data, df2_data)
        return LagResult(lag=lag, method="resample_1s"), af

    raise ValueError(
        f"compute_reference_lag: no reference channel pair available for "
        f"record={record.filename!r}, source={source!r} (tried: {tried}) — "
        "this indicates a problem in the data files, not a normal fallback case."
    )


def _interpolate_onto(
    reference_df: pd.DataFrame,
    reference_channel: str,
    other_df: pd.DataFrame,
    other_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate *other_df[other_key]* onto *reference_df*'s timestamps.

    Both series are restricted to their overlapping time window.

    Returns
    -------
    tuple of numpy.ndarray
        ``(actual, predicted)`` — reference values and interpolated other
        values, aligned one-to-one over the overlap window.
    """
    ref = reference_df[["timestamp", reference_channel]].dropna()
    oth = other_df[["timestamp", other_key]].dropna()

    origin = min(ref["timestamp"].iloc[0], oth["timestamp"].iloc[0])
    x_ref = ((ref["timestamp"] - origin) / pd.Timedelta(seconds=1)).to_numpy()
    x_oth = ((oth["timestamp"] - origin) / pd.Timedelta(seconds=1)).to_numpy()

    overlap_start = max(x_ref[0], x_oth[0])
    overlap_end = min(x_ref[-1], x_oth[-1])
    mask = (x_ref >= overlap_start) & (x_ref <= overlap_end)

    actual = ref[reference_channel].to_numpy(dtype=float)[mask]
    predicted = np.interp(x_ref[mask], x_oth, oth[other_key].to_numpy(dtype=float))
    return actual, predicted


def _convert_units(values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert *values* from *from_unit* to *to_unit* via pint.

    Raises
    ------
    ValueError
        If either unit string is unparseable, or the two units are not
        dimensionally compatible (e.g. volt to ampere) — this indicates a
        problem in the defs files, not a normal fallback case.
    """
    ureg = pint.get_application_registry()
    try:
        return (values * ureg.Unit(from_unit)).to(ureg.Unit(to_unit)).magnitude
    except (pint.UndefinedUnitError, pint.DimensionalityError) as e:
        raise ValueError(
            f"_convert_units: cannot convert {from_unit!r} to {to_unit!r}: {e}"
        ) from e


def compare_field(
    field: AliasedField,
    source: str,
    pigbrother_group_df: pd.DataFrame,
    pupitre_corrected_df: pd.DataFrame,
    compute_dtw: bool = False,
    plot: bool = False,
    output_dir: str | None = None,
    show: bool = False,
    downsample_percent: float = 100.0,
    reference_lag: LagResult | None = None,
) -> FieldComparisonResult:
    """Compare one field's timeseries between pigbrother and lag-corrected pupitre.

    Parameters
    ----------
    field : AliasedField
        Field to compare. When ``field.pupitre_unit`` and
        ``field.pigbrother_unit`` are both declared and differ (e.g.
        ``"tesla"`` vs. ``"millitesla"``), the pigbrother values are
        converted to the pupitre unit (pupitre is the reference) before
        metrics are computed and before plotting — see :func:`_convert_units`.
    source : str
        Pigbrother source name (``"overview"`` or ``"archive"``), used for
        labeling only — the reference lag correction has already been
        applied to *pupitre_corrected_df* by the caller
        (see :func:`compute_reference_lag`).
    pigbrother_group_df : pandas.DataFrame
        Loaded pigbrother data for *field.pigbrother_group*.
    pupitre_corrected_df : pandas.DataFrame
        Pupitre data with the reference lag already applied.
    compute_dtw : bool, optional
        Whether to also compute DTW distance (skipped above
        ``_DTW_MAX_POINTS`` regardless, since it is slow).
    plot : bool, optional
        Whether to render/save a comparison overlay plot.
    output_dir : str, optional
        Directory to save the plot in, when *plot* is True.
    show : bool, optional
        Display the plot interactively.
    downsample_percent : float, optional
        Percentage of points to plot (see
        :func:`~python_magnetrun.analysis.plotting.plot_comparison`).
    reference_lag : LagResult, optional
        The reference lag the caller applied to *pupitre_corrected_df*
        (see :func:`compute_reference_lag`), used only to label the plot's
        filename and title with the lag method and value. ``None`` (e.g.
        ``lag_method="none"``) labels the plot as ``lag_method=none``.

    Returns
    -------
    FieldComparisonResult

    Raises
    ------
    ValueError
        Propagated from :func:`_convert_units` when *field.pupitre_unit*
        and *field.pigbrother_unit* are both declared but unparseable or
        dimensionally incompatible — this indicates a problem in the defs
        files, not a normal fallback case.
    """
    if (
        pigbrother_group_df.empty
        or field.pigbrother_channel not in pigbrother_group_df.columns
    ):
        return FieldComparisonResult(
            field=field,
            source=source,
            available=False,
            reason=(
                f"pigbrother channel {field.pigbrother_channel!r} not found in "
                f"{source}/{field.pigbrother_group}"
            ),
        )
    if (
        pupitre_corrected_df.empty
        or field.pupitre_key not in pupitre_corrected_df.columns
    ):
        return FieldComparisonResult(
            field=field,
            source=source,
            available=False,
            reason=f"pupitre channel {field.pupitre_key!r} not found",
        )

    actual, predicted = _interpolate_onto(
        pigbrother_group_df,
        field.pigbrother_channel,
        pupitre_corrected_df,
        field.pupitre_key,
    )
    if len(actual) < 2:
        return FieldComparisonResult(
            field=field,
            source=source,
            available=False,
            reason="no overlapping time window between pigbrother and pupitre",
        )

    needs_unit_conversion = bool(
        field.pupitre_unit
        and field.pigbrother_unit
        and field.pupitre_unit != field.pigbrother_unit
    )
    if needs_unit_conversion:
        logger.warning(
            f"compare_field: {field.pupitre_key} unit mismatch — converting pigbrother "
            f"{field.pigbrother_channel!r} from {field.pigbrother_unit!r} to pupitre unit "
            f"{field.pupitre_unit!r}"
        )
        actual = _convert_units(actual, field.pigbrother_unit, field.pupitre_unit)

    metrics = compare_series(
        actual, predicted, compute_dtw=compute_dtw and len(actual) <= _DTW_MAX_POINTS
    )

    plot_path = None
    if plot:
        lag_method_str = reference_lag.method if reference_lag is not None else "none"
        lag_value_str = (
            f"{reference_lag.seconds:.3f}s" if reference_lag is not None else "n/a"
        )

        pigbrother_plot_df = pigbrother_group_df
        if needs_unit_conversion:
            pigbrother_plot_df = pigbrother_group_df.copy()
            pigbrother_plot_df[field.pigbrother_channel] = _convert_units(
                pigbrother_plot_df[field.pigbrother_channel].to_numpy(dtype=float),
                field.pigbrother_unit,
                field.pupitre_unit,
            )

        pint_unit = (
            pint.get_application_registry().Unit(field.pupitre_unit)
            if field.pupitre_unit
            else None
        )
        ylabel = format_axis_label(field.pupitre_symbol or field.pupitre_key, pint_unit)

        output_path = (
            f"{output_dir}/{field.pupitre_key}_{source}_{lag_method_str}_comparison.png"
            if output_dir
            else None
        )
        plot_comparison(
            pigbrother_plot_df,
            pupitre_corrected_df,
            x_col="timestamp",
            y_col1=field.pigbrother_channel,
            y_col2=field.pupitre_key,
            label1=f"pigbrother/{source}",
            label2="pupitre",
            title=(
                f"{field.pupitre_key} vs {field.pigbrother_group}/{field.pigbrother_channel} "
                f"({source}, lag_method={lag_method_str}, lag={lag_value_str})"
            ),
            ylabel=ylabel,
            downsample_percent=downsample_percent,
            show=show,
            save=bool(output_path),
            output_path=output_path,
        )
        plot_path = output_path

    return FieldComparisonResult(
        field=field,
        source=source,
        available=True,
        n_points=len(actual),
        metrics=metrics,
        plot_path=plot_path,
    )


def compare_all_fields(
    record: OverviewRecord,
    fields: list[AliasedField] | None = None,
    sources: tuple[str, ...] = PIGBROTHER_SOURCES,
    lag_method: Literal["resample_1s", "interpolated", "none"] = "resample_1s",
    compute_dtw: bool = False,
    plot: bool = False,
    output_dir: str | None = None,
    show: bool = False,
) -> dict[str, dict[str, FieldComparisonResult]]:
    """Compare every pupitre/pigbrother aliased field for *record*.

    For each *source* in *sources*, a single reference lag is computed from
    ``Idcct1``/``Courant_A1`` (or the ``Idcct3``/``Courant_A3`` fallback) and
    applied to pupitre data once; every field is then compared against that
    one lag-corrected pupitre DataFrame — no further lag computation is done
    per field.

    Parameters
    ----------
    record : OverviewRecord
        Record to compare (from
        :func:`~python_magnetrun.analysis.processing.process_overview_file`).
    fields : list of AliasedField, optional
        Fields to compare. Defaults to every field returned by
        :func:`discover_pupitre_pigbrother_fields`.
    sources : tuple of str, optional
        Pigbrother sources to compare against. Default: ``("overview", "archive")``.
    lag_method : {"resample_1s", "interpolated", "none"}, optional
        Reference-lag algorithm (see :func:`compute_reference_lag`). ``"none"``
        skips reference-lag computation entirely and compares pupitre data
        against pigbrother as-is, with no shift applied. ``"resample_1s"``
        combined with ``source="archive"`` is also skipped (with a logged
        warning) rather than computed: that method assumes a 1 Hz native
        rate and silently misinterprets the lag against 120 Hz Archive data
        (see :func:`~python_magnetrun.analysis.synchronization.compute_lag_interpolated`'s
        docstring) — use ``"interpolated"`` for reliable Archive lags.
    compute_dtw : bool, optional
        Whether to also compute DTW distance per field (slower).
    plot : bool, optional
        Whether to render/save comparison overlay plots.
    output_dir : str, optional
        Directory to save plots in, when *plot* is True.
    show : bool, optional
        Display plots interactively.

    Returns
    -------
    dict
        ``{pupitre_key: {source: FieldComparisonResult}}``. A source with no
        data files at all for *record* (e.g. no paired Archive file) is
        omitted rather than raising — that is a normal, expected situation.
        Fields whose ``(pigbrother_group, pigbrother_channel)`` is absent
        from *record*'s overview TDMS structure are dropped entirely (no
        key at all) rather than appearing with ``available=False``, unless
        a merged tap channel covers them (see :func:`_find_combined_tap_channel`),
        in which case the field is compared against that merged channel instead.

    Raises
    ------
    ValueError
        Propagated from :func:`compute_reference_lag` when data files are
        present but neither reference pair's columns can be found — this
        indicates a problem in the data files.
    """
    all_fields = discover_pupitre_pigbrother_fields()
    reference_fields = {
        af.pupitre_key: af for af in all_fields if af.pupitre_key in REFERENCE_LAG_KEYS
    }
    fields = fields if fields is not None else all_fields

    overview_channels = _load_overview_channel_map(record)
    if overview_channels:
        kept, dropped, redirected = [], [], []
        for af in fields:
            available = overview_channels.get(af.pigbrother_group, set())
            if af.pigbrother_channel in available:
                kept.append(af)
                continue
            merged_channel = _find_combined_tap_channel(
                af.pigbrother_channel, available
            )
            if merged_channel is not None:
                kept.append(replace(af, pigbrother_channel=merged_channel))
                redirected.append((af, merged_channel))
            else:
                dropped.append(af)
        if redirected:
            logger.info(
                f"compare_all_fields: redirecting fields to merged tap channels: "
                f"{[f'{af.pupitre_key}({af.pigbrother_group}/{af.pigbrother_channel} -> {merged})' for af, merged in redirected]}"
            )
        if dropped:
            logger.info(
                f"compare_all_fields: dropping fields absent from overview TDMS "
                f"structure: {[f'{af.pupitre_key}({af.pigbrother_group}/{af.pigbrother_channel})' for af in dropped]}"
            )
        fields = kept

    pupitre_keys = sorted({af.pupitre_key for af in fields} | set(REFERENCE_LAG_KEYS))
    pupitre_df = _load_pupitre_fields(record, pupitre_keys)

    pigbrother_cache: dict[tuple[str, str], pd.DataFrame] = {}
    results: dict[str, dict[str, FieldComparisonResult]] = {
        af.pupitre_key: {} for af in fields
    }

    for source in sources:
        ref_group_df = _load_pigbrother_group(
            record, source, "Courants_Alimentations", pigbrother_cache
        )
        if pupitre_df.empty or ref_group_df.empty:
            logger.warning(
                f"compare_all_fields: no data for source={source!r}, skipping"
            )
            continue

        if lag_method == "none":
            reference_lag, reference_field = None, None
            pupitre_corrected = pupitre_df
            logger.info(
                f"compare_all_fields: lag_method='none' for source={source!r} — "
                "comparing pupitre data with no lag correction applied"
            )
        elif source == "archive" and lag_method == "resample_1s":
            reference_lag, reference_field = None, None
            pupitre_corrected = pupitre_df
            logger.warning(
                "compare_all_fields: lag_method='resample_1s' is unreliable for "
                "source='archive' — it treats each cross-correlation index step as "
                "1 second, which is only correct at a 1 Hz native rate (Overview); "
                "against 120 Hz Archive data it silently misinterprets the lag (see "
                "compute_lag_interpolated's docstring). Skipping lag computation for "
                "this source — comparing pupitre data with no lag correction applied. "
                "Use lag_method='interpolated' instead for reliable Archive lags."
            )
        else:
            reference_lag, reference_field = compute_reference_lag(
                record,
                source,
                pupitre_df,
                ref_group_df,
                reference_fields,
                lag_method=lag_method,
            )
            logger.info(
                f"compare_all_fields: reference lag for source={source!r}: {reference_lag} "
                f"(field={reference_field.pupitre_key}/{reference_field.pigbrother_channel})"
            )
            pupitre_corrected = apply_lag_correction(pupitre_df, reference_lag.lag)

        for af in fields:
            group_df = _load_pigbrother_group(
                record, source, af.pigbrother_group, pigbrother_cache
            )
            result = compare_field(
                af,
                source,
                group_df,
                pupitre_corrected,
                compute_dtw=compute_dtw,
                plot=plot,
                output_dir=output_dir,
                show=show,
                reference_lag=reference_lag,
            )
            result.reference_lag = reference_lag
            result.reference_field = reference_field
            results[af.pupitre_key][source] = result

    return results


def print_comparison_summary(
    results: dict[str, dict[str, FieldComparisonResult]],
) -> None:
    """Log one table per pigbrother source, titled with its reference lag.

    Each pigbrother source (``"overview"``, ``"archive"``, ...) shares a
    single reference lag computed from one pupitre/pigbrother field pair
    (see :func:`compute_reference_lag`); that lag, the lag method, and the
    field pair are shown in a title line above each source's table, or a
    "no lag correction applied" note when ``lag_method="none"`` was used.
    Each row also lists the pigbrother alias (``group/channel``) that the
    pupitre key was compared against, as declared in ``pupitre-defs.json``.

    Parameters
    ----------
    results : dict
        Output of :func:`compare_all_fields`.
    """
    from tabulate import tabulate

    by_source: dict[str, list[tuple[str, FieldComparisonResult]]] = {}
    for pupitre_key, results_for_key in results.items():
        for source, result in results_for_key.items():
            by_source.setdefault(source, []).append((pupitre_key, result))

    headers = [
        "pupitre_key",
        "pigbrother_alias",
        "n",
        "correlation",
        "mape [%]",
        "euclidean",
        "rmse",
        "mae",
        "note",
    ]
    for source, entries in by_source.items():
        ref_lag, ref_field = next(
            (
                (r.reference_lag, r.reference_field)
                for _, r in entries
                if r.reference_lag is not None
            ),
            (None, None),
        )
        if ref_lag is not None and ref_field is not None:
            title = (
                f"source={source!r}  reference_lag={ref_lag.seconds:.3f}s  "
                f"lag_method={ref_lag.method}  "
                f"field={ref_field.pupitre_key}/{ref_field.pigbrother_channel}"
            )
        else:
            title = f"source={source!r}  lag_method=none (no lag correction applied)"

        rows = []
        for pupitre_key, result in entries:
            alias = f"{result.field.pigbrother_group}/{result.field.pigbrother_channel}"
            if not result.available:
                rows.append([pupitre_key, alias, "-", "-", "-", "-", "-", "-", result.reason])
                continue
            dist = result.metrics["distances"]
            rows.append(
                [
                    pupitre_key,
                    alias,
                    result.n_points,
                    f"{dist.correlation:.3f}",
                    f"{dist.mape:.2f}",
                    f"{dist.euclidean:.3f}",
                    f"{dist.rmse:.3f}",
                    f"{dist.mae:.3f}",
                    "",
                ]
            )

        print(f"{title}\n" + tabulate(rows, headers=headers, tablefmt="simple"))
