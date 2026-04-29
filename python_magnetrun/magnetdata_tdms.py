"""TdmsMagnetData — TDMS-backed magnet data (pigbrother files)."""

from __future__ import annotations

import contextlib
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .utils.downsampling import DownsampleConfig

import pandas as pd
import pytz

from .magnetdata_base import DataType, MagnetDataBase
from .utils.timestamps import parse_filename_timestamp
from .utils.timezone import (
    local_to_utc_naive,
    series_utc_to_local_naive,
    timerange_to_utc,
)

logger = logging.getLogger(__name__)


class _LazyGroupDict(dict):
    """Dict that loads TDMS groups on demand via ``_ensure_group_loaded``."""

    def __init__(self, owner: TdmsMagnetData) -> None:
        super().__init__()
        self._owner = owner

    def __getitem__(self, key: str) -> pd.DataFrame:
        self._owner._ensure_group_loaded(key)
        return super().__getitem__(key)


class TdmsMagnetData(MagnetDataBase):
    """TDMS-backed magnet data.

    ``self.Data`` is a ``dict[str, pd.DataFrame]`` keyed by group name.
    ``self.Type`` is :attr:`DataType.TDMS`.

    ``start_timestamp`` is set in three phases:

    1. At construction: parsed from the filename (local time, lightweight).
    2. :meth:`_validate_start_timestamp` calls :meth:`_apply_wf_timestamps`,
       which reads the accurate UTC ``wf_start_time`` from every group's channel
       properties and overwrites ``start_timestamp`` (and sets ``end_timestamp``)
       when all groups agree.
    3. The settled timestamp is converted to a naive UTC :class:`~datetime.datetime`.
    """

    def __init__(
        self,
        filename: str,
        Groups: dict,
        Keys: list[str],
        Data: dict | None = None,
        defs_file: str | None = None,
        time_zone: str = "Europe/Paris",
        _tdms_file: Any = None,
        _tdms_groups: dict | None = None,
    ) -> None:
        # Initialise backing store before super().__init__ so that the Data
        # property is valid if any base-class code accesses it.
        lazy: _LazyGroupDict = _LazyGroupDict(self)
        if isinstance(Data, dict):
            lazy.update(Data)
        self._data: _LazyGroupDict = lazy
        # Lazy-loading state: file handle and per-group TdmsGroup objects.
        # Groups are loaded on first access via _ensure_group_loaded().
        self._tdms_file: Any = _tdms_file
        self._tdms_groups: dict = _tdms_groups or {}
        super().__init__(filename, Groups, Keys, defs_file=defs_file)
        dt = parse_filename_timestamp(filename)
        self.start_timestamp = pd.Timestamp(dt) if dt is not None else None
        self._validate_start_timestamp()
        # Convert to naive UTC datetime — wf_start_time is already UTC-aware;
        # filename-derived timestamps are local and need tz_localize first.
        if self.start_timestamp is not None or self.end_timestamp is not None:
            if self.start_timestamp is not None:
                self.start_timestamp = local_to_utc_naive(
                    self.start_timestamp, time_zone
                )
            if self.end_timestamp is not None:
                self.end_timestamp = local_to_utc_naive(self.end_timestamp, time_zone)

    # --- lazy group loading ------------------------------------------

    def _ensure_group_loaded(self, gname: str) -> None:
        """Load *gname*'s DataFrame from disk on first access.

        Reads the group via ``TdmsGroup.as_dataframe()``, renames columns
        (spaces → underscores), validates ``wf_samples``, and stores the
        result in ``self.Data[gname]``.  Subsequent calls for the same group
        are no-ops.
        """
        if gname in self.Data:
            return
        if gname not in self._tdms_groups:
            raise RuntimeError(
                f"TdmsMagnetData._ensure_group_loaded: group {gname!r} not found "
                f"in {self.FileName!r}. Available: {list(self._tdms_groups)}"
            )
        group = self._tdms_groups[gname]
        df = group.as_dataframe(time_index=False, absolute_time=False, scaled_data=True)
        df.rename(
            columns={col: col.replace(" ", "_") for col in df.columns},
            inplace=True,
        )
        if self.Groups.get(gname):
            first_ch = next(iter(self.Groups[gname]))
            expected = self.Groups[gname][first_ch].get("wf_samples")
            if expected is None:
                for ch in self.Groups[gname]:
                    self.Groups[gname][ch]["wf_samples"] = len(df)
            elif len(df) != expected:
                logger.warning(
                    "group %r: loaded %d rows but wf_samples=%d — updating",
                    gname,
                    len(df),
                    expected,
                )
                for ch in self.Groups[gname]:
                    self.Groups[gname][ch]["wf_samples"] = len(df)
        assert isinstance(self.Data, dict)
        self.Data[gname] = df
        logger.debug("_ensure_group_loaded: loaded group %r (%d rows)", gname, len(df))

    # --- Data property -----------------------------------------------

    @property
    def Data(self) -> _LazyGroupDict:
        return self._data

    @Data.setter
    def Data(self, value: dict) -> None:
        if isinstance(value, _LazyGroupDict):
            self._data = value
        else:
            new_lazy: _LazyGroupDict = _LazyGroupDict(self)
            new_lazy.update(value)
            self._data = new_lazy

    # --- resource lifecycle ------------------------------------------

    def close(self) -> None:
        """Release the open TDMS file handle."""
        if self._tdms_file is not None:
            with contextlib.suppress(Exception):
                self._tdms_file.close()
            self._tdms_file = None

    def __del__(self) -> None:
        self.close()

    def _validate_start_timestamp(self) -> None:
        """Refine ``start_timestamp`` using TDMS ``wf_start_time`` channel properties.

        Calls :meth:`_apply_wf_timestamps`.  When no ``wf_start_time`` is
        available the filename-derived ``start_timestamp`` is preserved.
        """
        self._apply_wf_timestamps()

    def _apply_wf_timestamps(self) -> None:
        """Refine start/end timestamps from TDMS ``wf_start_time`` channel properties.

        Iterates over all non-``Infos`` groups and collects ``wf_start_time``
        from the first channel of each group.  If all groups report the same
        value, ``self.start_timestamp`` is overwritten with that value and
        ``self.end_timestamp`` is computed from the first group's duration
        (``wf_increment × wf_samples``).

        When groups disagree a warning is logged and neither timestamp is
        changed, so the filename-derived ``start_timestamp`` is preserved.
        """
        group_starts: dict[str, datetime] = {}

        for gname, channels in self.Groups.items():
            if gname == "Infos":
                continue
            if not isinstance(channels, dict):
                continue
            first_channel = next(
                (k for k, v in channels.items() if isinstance(v, dict)), None
            )
            if first_channel is None:
                continue
            props = channels[first_channel]
            if "wf_start_time" not in props:
                continue
            try:
                group_starts[gname] = props["wf_start_time"].astype(datetime)
            except (TypeError, ValueError):
                logger.warning(
                    f"_apply_wf_timestamps: could not convert wf_start_time in group {gname!r} of {self.FileName!r}"
                )

        if not group_starts:
            return

        reference = next(iter(group_starts.values()))
        inconsistent = [g for g, t in group_starts.items() if t != reference]

        if inconsistent:
            first_gname = next(iter(group_starts))
            logger.warning(
                f"_apply_wf_timestamps: wf_start_time is inconsistent across groups "
                f"in {self.FileName!r} — differing groups: {inconsistent} "
                f"(reference from {first_gname!r}: {reference}); "
                f"keeping filename-derived start_timestamp"
            )
            return

        # All groups agree — overwrite start_timestamp with the accurate UTC value.
        # wf_start_time from LabVIEW TDMS is UTC; attach UTC tzinfo so the
        # conversion step in __init__ can distinguish it from a local filename timestamp.
        ref_utc = (
            reference if reference.tzinfo is not None else pytz.utc.localize(reference)
        )
        self.start_timestamp = pd.Timestamp(ref_utc)

        # Derive end_timestamp from the first group's duration.
        first_gname = next(iter(group_starts))
        first_ch = next(
            (k for k, v in self.Groups[first_gname].items() if isinstance(v, dict)),
            None,
        )
        if first_ch is not None:
            props = self.Groups[first_gname][first_ch]
            dt = props.get("wf_increment", 0)
            samples = props.get("wf_samples", 0)
            self.end_timestamp = self.start_timestamp + pd.Timedelta(
                seconds=dt * samples
            )

    @property
    def Type(self) -> DataType:
        return DataType.TDMS

    # --- core data access --------------------------------------------

    def getTdmsData(self, group: str, channel: str | list[str] | None) -> pd.DataFrame:
        if not isinstance(self.Data, dict):
            raise Exception(
                f"MagnetData/getTdmsData: {self.FileName} - expect Data to be a dict"
            )
        self._ensure_group_loaded(group)
        if channel is None or not channel:
            return self.Data[group]
        return self.Data[group][channel]

    def getData(
        self,
        key: list[str] | str | None = None,
        downsample: DownsampleConfig | None = None,
    ) -> pd.DataFrame:
        from .utils.downsampling import downsample_dataframe

        logger.debug(f"getData: key={key}, downsample={downsample}")

        channels: list[str] = []
        groups: list[str] = []

        if isinstance(key, str):
            if "/" in key:
                (group, channel) = key.split("/")
                channels.append(channel)
            else:
                group = key
            groups.append(group)

        elif isinstance(key, list):
            for item in key:
                if "/" in item:
                    (group, channel) = item.split("/")
                    channels.append(channel)
                    if group not in groups:
                        groups.append(group)
                else:
                    groups.append(item)

        if groups and len(groups) > 1:
            groups = list(dict.fromkeys(groups))

        if len(groups) == 0 or len(groups) > 1:
            raise RuntimeError(
                f"magnetata:getData for tdms - expect only one group - got {len(groups)}: {groups}"
            )

        df = self.getTdmsData(groups[0], channels)
        if downsample is not None and len(df) > downsample.n_out:
            time_col = "t" if "t" in df.columns else df.columns[0]
            value_cols = [c for c in df.columns if c != time_col]
            df = downsample_dataframe(
                df, time_col=time_col, value_cols=value_cols, config=downsample
            )

        # Attach unit metadata so plotting functions can label axes correctly.
        # Uses a per-key try/except because Units() may not have been called yet.
        units_attrs: dict = {}
        for col in df.columns:
            with contextlib.suppress(KeyError, RuntimeError):
                units_attrs[col] = self.getUnitKey(col)
        df.attrs["units"] = units_attrs

        return df

    def getKeys(self) -> list[str]:
        return self.Keys

    # --- units -------------------------------------------------------

    def PigBrotherUnits(self, key: str, debug: bool = False) -> tuple:  # noqa: N802
        from pint import UnitRegistry

        logger.debug(f"PigBrotherUnits: key={key}")
        ureg: UnitRegistry = UnitRegistry()

        _pig_units = {
            "Courant": ("I", ureg.ampere),
            "Tension": ("U", ureg.volt),
            "Puissance": ("Power", ureg.watt),
            "Power": ("Power", ureg.watt),
            "Champ_magn": ("B", ureg.gauss),
        }

        for entry in _pig_units:
            if entry in key:
                return _pig_units[entry]

        return ()

    def Units(
        self, debug: bool = False, json_file: str | None = None
    ) -> None:  # noqa: N802
        """Populate ``self.units``.

        Resolution order:

        1. *json_file* argument (explicit override) / ``self.defs_file``
           — if a key's embedded TDMS ``unit_string`` disagrees, a warning is
           printed and the defs_file value is used.
        2. Embedded ``unit_string`` from TDMS channel properties (for keys not
           covered by the defs_file).
        3. ``PigBrotherUnits`` keyword matching (final fallback).
        """
        from pint.errors import UndefinedUnitError

        from .magnetdata_base import _make_ureg

        ureg = _make_ureg()

        # Step 1 — collect embedded unit strings from TDMS channel properties
        tdms_unit_strs: dict[str, str] = {}
        for gname, channels in self.Groups.items():
            if gname == "Infos" or not isinstance(channels, dict):
                continue
            for cname, props in channels.items():
                raw = props.get("unit_string", "") if hasattr(props, "get") else ""
                logger.debug(
                    f"Units: group={gname!r}, channel={cname!r}, raw unit_string={raw!r}"
                )
                if raw and raw.strip():
                    tdms_unit_strs[f"{gname}/{cname}"] = raw.strip()
        logger.debug(f"Units: collected TDMS unit strings: {tdms_unit_strs}")

        # Step 2 — load defs_file (takes priority); warn on unit mismatch
        resolved = json_file or self.defs_file
        logger.debug(
            f"Units: json_file={json_file!r}, defs_file={self.defs_file!r}, resolved={resolved!r}"
        )
        if resolved is not None:
            from .field_defs import load_defs

            field_defs = load_defs(resolved)
            add_fields = []
            for key, tdms_unit_str in tdms_unit_strs.items():
                if key not in field_defs:
                    if key.startswith("_"):
                        continue
                    logger.debug(
                        f"Units: TDMS key {key!r} not found in defs_file {resolved!r}"
                    )
                    add_fields.append(key)
                    continue
                defs_unit_str = field_defs[key].get("unit")
                if defs_unit_str is not None and defs_unit_str != tdms_unit_str:
                    logger.warning(
                        f"Units: {key} — TDMS embedded unit {tdms_unit_str!r} differs from "
                        f"defs_file unit {defs_unit_str!r}; overriding with defs_file value -- aka {defs_unit_str!r}"
                    )

            self.load_units_from_json(resolved, debug=debug)

        # Step 3 — use embedded TDMS unit_string for keys not set by defs_file
        for key, unit_str in tdms_unit_strs.items():
            logger.debug(
                f"step3: Units: processing TDMS unit for key {key!r}: {unit_str!r}"
            )
            if key in self.units:
                continue
            if key.endswith("/t"):
                self.units[key] = ("t", ureg.second)
                continue
            elif key.endswith("/timestamp"):
                self.units[key] = ("time", None)
                continue
            try:
                logger.debug(
                    f"Units: parsing TDMS unit string {unit_str!r} for key {key!r}"
                )
                pint_unit = ureg.parse_expression(unit_str)
                self.units[key] = (key.split("/")[-1], pint_unit)
            except (ValueError, AttributeError, UndefinedUnitError):
                logger.debug(
                    f"Units: cannot parse TDMS unit {unit_str!r} for {key}, falling back to keyword match"
                )

        # Step 4 — PigBrotherUnits keyword matching for any remaining entries
        for entry in self.Data:
            logger.debug(
                f"step4: Units: processing PigBrotherUnits for entry {entry!r}"
            )
            if not isinstance(entry, str):
                continue
            if entry in self.units:
                continue
            if entry == "t":
                self.units["t"] = ("t", ureg.second)
            else:
                group = entry
                if "/" in entry:
                    (group, channel) = entry.split("/")
                    if channel == "t":
                        self.units[entry] = ("t", ureg.second)
                    elif channel == "timestamp":
                        self.units[entry] = ("time", None)
                    continue
                pig = self.PigBrotherUnits(group)
                if pig:
                    logger.debug(
                        f"Units: overwrite {entry!r} with PigBrotherUnits {pig}"
                    )
                    self.units[entry] = pig

        if debug:
            logger.debug(f"Units: {self.Keys}")

    def getUnitKey(self, key: str) -> tuple:
        logger.debug(f"getUnitKey: key={key}")
        if key not in self.Keys:
            from pint import UnitRegistry

            ureg: UnitRegistry = UnitRegistry()
            if key == "t":
                return ("t", ureg.second)
            elif key == "timestamp":
                return ("time", None)
            else:
                raise RuntimeError(
                    f"{key} not defined in data - available keys are {self.Keys}"
                )
        if key in self.units:
            return self.units[key]

        (group, channel) = key.split("/")
        logger.debug(
            f"getUnitKey: key={key} - group={group}, channel={channel} (in_units:{key in self.units}, in_Keys:{key in self.Keys})"
        )
        return self.PigBrotherUnits(channel)

    def renameData(self, columns: dict) -> None:  # noqa: N802
        """TDMS data does not support renaming channels.

        Emits a warning when *columns* is non-empty so callers are not silently
        misled into thinking the rename was applied.
        """
        if columns:
            logger.warning(
                f"renameData: TDMS does not support channel renaming; columns={list(columns)} ignored"
            )

    def addTime(self, time_zone: str = "Europe/Paris") -> int:  # noqa: N802
        """Implement the MagnetDataBase.addTime contract for TDMS data.

        Eagerly computes both ``t`` and ``timestamp`` (naive UTC) for **all**
        non-``Infos`` groups.  Call this once before :meth:`extractTimeData`
        or any ``timestamp``-based plot.

        :param time_zone: accepted for interface compatibility; TDMS timestamps
            are derived from ``wf_start_time`` which is already UTC, so no
            timezone conversion is required here.
        """
        self.addTdmsTime()
        self.addTdmsTimestamp()  # timezone=None → stores naive UTC
        return 0

    def cleanupData(  # noqa: N802
        self,
        keys_to_remove: list[str] | None = None,
        keys_to_rename: dict[str, str] | None = None,
        keys_to_add: dict[str, dict[str, Any]] | None = None,
        debug: bool = False,
    ) -> int:
        """Apply ETL operations to TDMS data.

        ``keys_to_add`` entries must use ``"Group/Channel"`` syntax consistent
        with :meth:`addData`.  ``keys_to_rename`` is ignored (TDMS does not
        support channel renaming).

        :param keys_to_remove: list of ``"Group/Channel"`` keys to drop.
        :param keys_to_rename: unused for TDMS; a warning is emitted if non-empty.
        :param keys_to_add: ``{"Group/Channel": field_def}`` pairs where each
            ``field_def`` dict contains ``formula``, ``symbol``, ``unit``,
            ``label``, and ``description`` keys; each entry is evaluated via
            :meth:`addData`.
        :param debug: passed through to :meth:`addData`.
        :return: 0 on success.
        """
        if keys_to_rename:
            logger.warning(
                f"cleanupData: TDMS does not support renaming channels; keys_to_rename={list(keys_to_rename)} ignored"
            )

        if keys_to_add:
            for key, field_def in keys_to_add.items():
                if key not in self.Keys:
                    self.addData(
                        key,
                        field_def["formula"],
                        symbol=field_def["symbol"],
                        unit=field_def["unit"],
                        label=field_def["label"],
                        description=field_def["description"],
                        debug=debug,
                    )
                else:
                    logger.debug(f"cleanupData: key {key!r} already exists, skipping")

        if keys_to_remove:
            assert isinstance(self.Data, dict)
            for key in keys_to_remove:
                if "/" not in key:
                    logger.warning(
                        f"cleanupData: skip non-TDMS key {key!r} (no '/' separator)"
                    )
                    continue
                group, channel = key.split("/", 1)
                self._ensure_group_loaded(group)
                if group in self.Data and channel in self.Data[group].columns:
                    self.Data[group].drop(columns=[channel], inplace=True)
                    if key in self.Keys:
                        self.Keys.remove(key)
                else:
                    logger.debug(
                        f"cleanupData: key {key!r} not found, skipping removal"
                    )

        return 0

    # --- compute / add -----------------------------------------------

    def addData(  # noqa: N802
        self,
        key: str,
        formula: str,
        symbol: str,
        unit: Any,  # pint.Unit | str | None
        label: str,
        description: str,
        debug: bool = False,
    ) -> int:
        from pint.errors import UndefinedUnitError

        from .magnetdata_base import FieldMeta, _make_ureg

        (group, channel) = key.split("/")
        logger.debug(f"add: key={key} - group={group}, channel={channel}")
        self._ensure_group_loaded(group)

        nformula = formula.replace(f"{group}/", "")

        import re

        match = re.findall(r"(\w+)/(\w+)", nformula)
        if match:
            for matched in match:
                logger.debug(f"matched={matched[0]}/{matched[1]}")
                self.Data[group][matched[1]] = self.Data[matched[0]][matched[1]]  # type: ignore[index]
                nformula = nformula.replace(f"{matched[0]}/", "")
        logger.debug(f"formula: {nformula}")

        try:
            self.Data[group].eval(nformula, inplace=True)  # type: ignore[index]
            self.Keys.append(key)

            first_key = list(self.Groups[group].keys())[0]
            first_props = self.Groups[group][first_key]
            unit_str = unit if isinstance(unit, str) else ""
            self.Groups[group][channel] = {
                "wf_increment": first_props["wf_increment"],
                "wf_start_time": first_props.get("wf_start_time"),
                "wf_samples": first_props.get("wf_samples", 0),
                "wf_start_offset": first_props.get("wf_start_offset", 0.0),
                "unit_string": unit_str,
            }

            if isinstance(unit, str) and unit:
                try:
                    ureg = _make_ureg()
                    parsed = ureg.parse_expression(unit)
                    pint_unit = parsed.units if hasattr(parsed, "units") else parsed
                except (ValueError, UndefinedUnitError):
                    pint_unit = None
            else:
                pint_unit = unit if unit else None  # empty string → None

            self.units[key] = (symbol, pint_unit)
            self.field_meta[key] = FieldMeta(
                symbol=symbol, unit=pint_unit, label=label, description=description
            )

        except pd.errors.UndefinedVariableError as error:
            raise RuntimeError(
                f"addData: {key}: {nformula} - failed for tdms {group} data - error={error}"
            ) from error

        return 0

    def computeData(  # noqa: N802
        self,
        method: Any,
        key: str,
        kparams: list,
        symbol: str,
        unit: Any,  # pint.Unit | str | None
        label: str,
        description: str,
        debug: bool = False,
    ) -> None:
        raise RuntimeError(
            f"computeData: key={key} not implemented for pigbrother file"
        )

    # --- time utilities ----------------------------------------------

    def getStartDate(self, group: str | None = None) -> tuple:  # noqa: N802
        if group is None:
            group = next(
                (g for g in self.Groups if g != "Infos" and self.Groups[g]),
                None,
            )
        if group is None:
            return ()

        channel = list(self.Groups[group].keys())[0]
        props = self.Groups[group][channel]
        if "wf_start_time" not in props:
            return ()

        start_t = props["wf_start_time"].astype(datetime)
        logger.debug(f"getStartDate: tdms start_t={start_t} (type={type(start_t)})")

        from datetime import timedelta

        duration_s = self.getDuration(group)
        end_t = start_t + timedelta(seconds=duration_s)

        dformat = "%Y.%m.%d"
        tformat = "%H:%M:%S"
        return (
            start_t.strftime(dformat),
            start_t.strftime(tformat),
            end_t.strftime(dformat),
            end_t.strftime(tformat),
        )

    def getDuration(self, group: str | None = None) -> float:  # noqa: N802
        if group is None:
            group = next(
                (g for g in self.Groups if g != "Infos" and self.Groups[g]),
                None,
            )
        channel = list(self.Groups[group].keys())[0]
        ordered_dict = self.Groups[group][channel]
        dt = ordered_dict["wf_increment"]
        samples = ordered_dict["wf_samples"]
        logger.debug(f"getDuration: group={group}, dt={dt}, samples={samples}")
        return float(dt * samples)

    def addTdmsTime(self, group: str | None = None) -> int:  # noqa: N802
        """Add a ``'t'`` column to group(s) in Data.

        Uses ``wf_increment`` and ``wf_start_offset`` from the first channel's
        properties to compute ``t = index * dt + t_offset``.
        """
        assert isinstance(self.Data, dict)

        if (
            group is not None
            and group not in self._tdms_groups
            and group not in self.Data
        ):
            raise RuntimeError(
                f"MagnetData/addTdmsTime {self.FileName}: group '{group}' not found"
            )

        # Use Groups (always populated from metadata) as the source of group names
        # so that unloaded groups are still processed.
        groups_to_process = (
            [group]
            if group is not None
            else [
                g
                for g in self.Groups
                if g != "Infos" and isinstance(self.Groups[g], dict)
            ]
        )
        logger.debug(f"addTdmsTime: groups_to_process={groups_to_process}")

        for gname in groups_to_process:
            if gname == "Infos":
                continue
            self._ensure_group_loaded(gname)
            if "t" in self.Data[gname].columns:
                logger.debug(
                    f"addTdmsTime: 't' already present in group '{gname}', skipping"
                )
                continue

            group_channels = self.Groups.get(gname, {})
            if not group_channels:
                logger.warning(
                    f"addTdmsTime: no channel properties found for group '{gname}', skipping"
                )
                continue

            first_channel = list(group_channels.keys())[0]
            props = group_channels[first_channel]

            if "wf_increment" not in props or "wf_start_offset" not in props:
                logger.warning(
                    f"addTdmsTime: missing wf_increment/wf_start_offset for '{gname}/{first_channel}', skipping"
                )
                continue

            dt = props["wf_increment"]
            t_offset = props["wf_start_offset"]
            self.Data[gname]["t"] = self.Data[gname].index * dt + t_offset

            key = f"{gname}/t"
            if key not in self.Keys:
                self.Keys.append(key)

            if "t" not in self.Groups[gname]:
                self.Groups[gname]["t"] = {
                    "wf_samples": props.get("wf_samples", 0),
                    "wf_increment": dt,
                    "wf_start_offset": t_offset,
                    "wf_start_time": props.get("wf_start_time"),
                    "unit_string": "s",
                }

        return 0

    def get_time_range(self) -> tuple:
        """Return ``(start_datetime, end_datetime)`` from TDMS wf_start_time + duration."""
        # Use the first available non-Infos group
        for gname, channels in self.Groups.items():
            if gname == "Infos":
                continue
            first_channel = next(
                (k for k, v in channels.items() if isinstance(v, dict)), None
            )
            if first_channel is None:
                continue
            props = channels[first_channel]
            if "wf_start_time" not in props:
                continue
            start_dt = props["wf_start_time"].astype(datetime)
            dt = props.get("wf_increment", 0)
            samples = props.get("wf_samples", 0)
            duration_s = dt * samples
            from datetime import timedelta

            end_dt = start_dt + timedelta(seconds=duration_s)
            return (start_dt, end_dt)
        raise RuntimeError(
            f"{self.__class__.__name__}.get_time_range: no usable group found in {self.FileName}"
        )

    # --- extract -----------------------------------------------------

    def extractData(self, keys: list[str]) -> pd.DataFrame:  # noqa: N802
        logger.debug(f"extractData: filename={self.FileName}, keys={keys}")
        groups: list[str] = []
        channels: list[str] = []
        dfs: list[pd.DataFrame] = []
        for item in keys:
            if item != "t":
                (group, channel) = item.split("/")
                df = self.getTdmsData(group, channel)
                dfs.append(df)
                groups.append(group)
                channels.append(channel)

        result = pd.concat(dfs, axis=1)
        if "t" in keys:

            def all_same_string(lst: list) -> bool:
                return len(set(lst)) <= 1

            if all_same_string(groups):
                group = groups[0]
                result["t"] = self.Data[group]["t"]  # type: ignore[index]
            else:
                raise RuntimeError(
                    f"extractData: keys={keys} - cannot add t column - groups are not the same: {groups}"
                )

        return result

    def extractDataThreshold(
        self, key: str, threshold: float
    ) -> pd.DataFrame:  # noqa: N802
        (group, channel) = key.split("/")
        self._ensure_group_loaded(group)
        return self.Data[group][channel].loc[self.Data[group][channel] >= threshold]  # type: ignore[index]

    def addTdmsTimestamp(  # noqa: N802
        self,
        group: str | None = None,
        timezone: str | None = None,
    ) -> int:
        """Add a ``'timestamp'`` column (absolute datetime) to group(s) in Data.

        Requires ``wf_start_time`` and ``wf_increment`` in channel properties.
        Calls ``addTdmsTime`` first to ensure ``'t'`` column exists.

        :param timezone: optional IANA timezone name (e.g. ``"Europe/Paris"``).
            When provided the timestamp column is converted from UTC to that
            timezone using :mod:`pytz`.
        """
        assert isinstance(self.Data, dict)

        if (
            group is not None
            and group not in self._tdms_groups
            and group not in self.Data
        ):
            raise RuntimeError(
                f"addTdmsTimestamp {self.FileName}: group '{group}' not found"
            )

        groups_to_process = (
            [group]
            if group is not None
            else [
                g
                for g in self.Groups
                if g != "Infos" and isinstance(self.Groups[g], dict)
            ]
        )

        for gname in groups_to_process:
            if gname == "Infos":
                continue
            self._ensure_group_loaded(gname)
            if "timestamp" in self.Data[gname].columns:
                logger.debug(
                    f"addTdmsTimestamp: 'timestamp' already in '{gname}', skipping"
                )
                continue

            group_channels = self.Groups.get(gname, {})
            if not group_channels:
                logger.warning(
                    f"addTdmsTimestamp: no channel props for '{gname}', skipping"
                )
                continue

            first_channel = list(group_channels.keys())[0]
            props = group_channels[first_channel]
            if "wf_start_time" not in props:
                logger.warning(
                    f"addTdmsTimestamp: no wf_start_time for '{gname}', skipping"
                )
                continue

            self.addTdmsTime(group=gname)

            start_dt = props["wf_start_time"].astype(datetime)
            self.Data[gname]["timestamp"] = pd.Timestamp(start_dt) + pd.to_timedelta(
                self.Data[gname]["t"], unit="s"
            )

            if timezone is not None:
                tz = pytz.timezone(timezone)
                self.Data[gname]["timestamp"] = (
                    self.Data[gname]["timestamp"]
                    .dt.tz_localize(pytz.utc)
                    .dt.tz_convert(tz)
                )

            key = f"{gname}/timestamp"
            if key not in self.Keys:
                self.Keys.append(key)

            if "timestamp" not in self.Groups[gname]:
                self.Groups[gname]["timestamp"] = {
                    "wf_samples": props["wf_samples"],
                    "wf_increment": props["wf_increment"],
                    "wf_start_offset": props["wf_start_offset"],
                    "wf_start_time": props["wf_start_time"],
                    "unit_string": "datetime",
                }

        return 0

    def extractTimeData(  # noqa: N802
        self, timerange: str, group: str | None = None, time_zone: str = "Europe/Paris"
    ) -> pd.DataFrame:
        """Return rows whose ``timestamp`` falls within *timerange*.

        :param timerange: ``"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"`` in local
            time (the ``time_zone`` timezone).  Both boundaries are inclusive.
        :param group: TDMS group name (required).
        :param time_zone: IANA timezone of the datetime strings in *timerange*
            (default ``"Europe/Paris"``).
        :raises RuntimeError: if *group* is ``None`` or :meth:`addTime` has not
            been called yet.
        """
        if group is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.extractTimeData: group is required for TDMS data"
            )
        assert isinstance(self.Data, dict)
        if "timestamp" not in self.Data[group].columns:
            raise RuntimeError(
                f"{self.__class__.__name__}.extractTimeData: call addTime() before extractTimeData()"
            )
        logger.debug(f"Select data from {timerange}")
        t_start, t_end = timerange_to_utc(timerange, time_zone)
        return self.Data[group][
            self.Data[group]["timestamp"].between(t_start, t_end, inclusive="both")
        ]

    # --- persist / display -------------------------------------------

    def saveData(self, keys: list[str], filename: str) -> int:  # noqa: N802
        dfs = []
        for key in keys:
            (group, channel) = key.split("/")
            dfs.append(self.getTdmsData(group, channel))
        df = pd.concat(dfs)
        df.to_csv(filename, sep="\t", index=False, header=True)
        return 0

    def plotData(  # noqa: N802
        self,
        x: str,
        y: str,
        ax: Any,
        alpha: float = 1,
        label: str | None = None,
        normalize: bool = False,
        offset: float = 0,
        time_zone: str = "Europe/Paris",
        color: str | None = None,
        marker: str | None = None,
        linestyle: str | None = None,
        markevery: int | None = None,
    ) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        logger.info(f"plotData: plotting {y} vs {x} from {self.FileName!r}")
        matplotlib.rcParams["text.usetex"] = True

        if x not in self.Keys + ["t", "timestamp"]:
            raise RuntimeError(
                f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no x={x} key (valid keys= {self.Keys})"
            )

        if y not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {y} key (valid keys: {self.Keys})"
            )

        (ysymbol, yunit) = self.getUnitKey(y)
        (ygroup, ychannel) = y.split("/")
        if "/" in x:
            (xgroup, xchannel) = x.split("/")
        else:
            xgroup = ygroup
            xchannel = x

        if xgroup != ygroup:
            raise RuntimeError(
                f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: xgroup={xgroup} != {ygroup}"
            )

        self._ensure_group_loaded(xgroup)
        df = self.Data[xgroup].copy()  # type: ignore[index]

        # Convert naive UTC timestamp → naive local time for display
        if xchannel == "timestamp":
            df["timestamp"] = series_utc_to_local_naive(df["timestamp"], time_zone)

        kwargs: dict = {
            "x": xchannel,
            "y": ychannel,
            "ax": ax,
            "alpha": alpha,
            "grid": False,
        }
        if color is not None:
            kwargs["color"] = color
        if marker is not None:
            kwargs["marker"] = marker
        if linestyle is not None:
            kwargs["linestyle"] = linestyle
        if markevery is not None:
            kwargs["markevery"] = markevery

        if normalize:
            ymax = abs(df[ychannel].max())
            df[ychannel] /= ymax
            kwargs["label"] = f"{label or ychannel} (norm with {ymax:.3e} {yunit:~P})"
        elif label is not None:
            kwargs["label"] = label

        df.plot(**kwargs)

        if yunit is not None:
            plt.ylabel(f"{ysymbol} [{yunit:~P}]")

        (xsymbol, xunit) = self.getUnitKey(x)
        if xunit is not None:
            plt.xlabel(f"{xsymbol} [{xunit:~P}]")

    def stats(self, key: str | None = None) -> pd.DataFrame | None:
        from tabulate import tabulate

        logger.info("magnetdata.stats")
        if key is not None:
            (group, channel) = key.split("/")
            self._ensure_group_loaded(group)
            if group in self.Data:
                if channel in self.Data[group]:  # type: ignore[index]
                    logger.info(
                        tabulate(
                            self.Data[group][channel].describe(),  # type: ignore[index]
                            headers="keys",
                            tablefmt="psql",
                        )
                    )
                    return self.Data[group][channel].describe()  # type: ignore[index]
                else:
                    raise RuntimeError(
                        f"magnetdata/stats: cannot find channel {channel}"
                    )
            else:
                raise RuntimeError(f"magnetdata/stats: cannot find group {group}")
        else:
            for group in list(self._tdms_groups):
                self._ensure_group_loaded(group)
                logger.info(f"stats[{group}]: ")
                df = self.Data[group].describe(include="all")  # type: ignore[index]
                logger.info(tabulate(df, headers="keys", tablefmt="psql"))
        return None

    def info(self) -> None:
        from collections import OrderedDict

        from tabulate import tabulate

        logger.info(f"magnetdata: {self.FileName}, Type={self.Type.name}")
        headers = [
            "Group",
            "Channel",
            "Samples",
            "Increment",
            "start_time",
            "start_offset",
        ]
        tables = []
        for group, values in self.Groups.items():
            for item in values:
                print(
                    f"info: group={group}, item={item}, values={values[item]}",
                    flush=True,
                )
                if isinstance(values[item], dict | OrderedDict):
                    table = [
                        group,
                        item,
                        values[item]["wf_samples"],
                        values[item]["wf_increment"],
                        values[item]["wf_start_time"],
                        values[item]["wf_start_offset"],
                    ]
                    tables.append(table)
        print(tabulate(tables, headers, tablefmt="simple"))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Type={self.Type!r}, Groups={self.Groups!r}, Keys={self.Keys!r}, Data={self.Data!r})"
