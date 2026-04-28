"""PandasMagnetData and thin pandas-backed subclasses."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .utils.downsampling import DownsampleConfig

import pandas as pd
from natsort import natsorted

from .magnetdata_base import DataType, MagnetDataBase
from .utils.timestamps import parse_filename_timestamp
from .utils.timezone import (
    local_to_utc_naive,
    series_local_to_utc_naive,
    series_utc_to_local_naive,
    timerange_to_utc,
)

logger = logging.getLogger(__name__)


def _dataframe_keys(df: pd.DataFrame) -> list[str]:
    """Return DataFrame column names normalized to ``list[str]``."""
    return [str(column) for column in df.columns.tolist()]


def _get_duplicate_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are exact duplicates of an earlier column."""
    duplicates: set[str] = set()
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            if col.equals(df.iloc[:, y]):
                duplicates.add(df.columns.values[y])
    return list(duplicates)


class PandasMagnetData(MagnetDataBase):
    """Pandas-backed magnet data (pupitre .txt, .csv, StringIO).

    ``self.Data`` is always a :class:`pandas.DataFrame`.
    ``self.Type`` is ``0`` unless overridden by a subclass.
    """

    _TYPE: DataType = DataType.PUPITRE  # overridden by EnsightMagnetData → ENSIGHT

    def __init__(
        self,
        filename: str,
        Groups: dict,
        Keys: list[str],
        Data: pd.DataFrame | None = None,
        defs_file: str | None = None,
        time_zone: str = "Europe/Paris",
        _read_kwargs: dict | None = None,
    ) -> None:
        super().__init__(filename, Groups, Keys, Data, defs_file)
        # Lazy-loading state.  _read_kwargs holds the pd.read_csv arguments
        # needed to reload the full file on first data access.
        self._data_loaded: bool = Data is not None and (
            not isinstance(Data, pd.DataFrame) or len(Data) > 1
        )
        self._read_kwargs: dict = _read_kwargs or {}
        dt = parse_filename_timestamp(filename)  # in local time
        self.start_timestamp = pd.Timestamp(dt) if dt is not None else None
        self._validate_start_timestamp()
        # Convert to UTC — use self.start_timestamp (may have been overridden by
        # _validate_start_timestamp with a value from the Date/Time data columns).
        if self.start_timestamp is not None:
            self.start_timestamp = local_to_utc_naive(self.start_timestamp, time_zone)

    # --- lazy loading ------------------------------------------------

    def __getattribute__(self, name: str):
        """Trigger lazy loading on first access to ``Data``."""
        if name == "Data":
            try:
                loaded = object.__getattribute__(self, "_data_loaded")
            except AttributeError:
                loaded = True
            if not loaded:
                object.__getattribute__(self, "_ensure_data_loaded")()
        return object.__getattribute__(self, name)

    def _ensure_data_loaded(self) -> None:
        """Load the full file from disk on first data access.

        Uses ``_read_kwargs`` stored at construction time to reproduce the
        original ``pd.read_csv`` call.  Subsequent calls are no-ops.
        """
        if self._data_loaded:
            return
        if not self._read_kwargs:
            return
        with open(self.FileName) as f:
            df = pd.read_csv(f, **self._read_kwargs)
        self._data_loaded = True  # set before assigning self.Data to avoid recursion
        self.Data = df
        self.Keys = _dataframe_keys(df)
        logger.debug(
            "_ensure_data_loaded: loaded %s (%d rows)", self.FileName, len(df)
        )

    # --- abstract property -------------------------------------------

    @property
    def Type(self) -> DataType:
        return self._TYPE

    # --- core data access --------------------------------------------

    def getPandasData(self, key: list[str] | str | None) -> pd.DataFrame:
        """Return Data or a selection by *key*."""
        self._ensure_data_loaded()
        if key is None:
            if not isinstance(self.Data, pd.DataFrame):
                raise RuntimeError(
                    f"MagnetData/Data: {self.FileName} - expect Data to be a pandas dataframe"
                )
            return self.Data  # type: ignore[return-value]
        selected_keys: list[str] = []
        if isinstance(key, list):
            selected_keys = key
        elif isinstance(key, str):
            selected_keys = [key]
        for item in selected_keys:
            if item not in self.Keys:
                raise KeyError(
                    f"MagnetData/Data({key}): {self.FileName}: cannot get data for key={item}: no such key"
                )
        return self.Data[selected_keys]  # type: ignore[index]

    def getData(
        self,
        key: list[str] | str | None = None,
        downsample: DownsampleConfig | None = None,
    ) -> pd.DataFrame:
        from .utils.downsampling import downsample_dataframe

        df = self.getPandasData(key)
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

    def Units(
        self, debug: bool = False, json_file: str | None = None
    ) -> None:  # noqa: N802
        """Populate ``self.units`` from column names.

        Resolution order:
        1. *json_file* argument (explicit override)
        2. ``self.defs_file`` set at construction time
        3. Built-in pattern matching (fallback, kept for backward compatibility)

        When a JSON file is resolved the pattern block is still applied for any
        key not present in the file, so partial JSON files work correctly.
        """
        from .magnetdata_base import _make_ureg

        resolved = json_file or self.defs_file
        if resolved is not None:
            self.load_units_from_json(resolved, debug=debug)

        ureg = _make_ureg()

        # For keys not populated from JSON fall back to legacy pattern matching.
        # Keys that match no pattern (e.g. 'Date', 'Time') are silently skipped.
        for key in self.Keys:
            if key in self.units:
                continue  # already populated from JSON
            else:
                logger.warning(
                    f"Units: no JSON definition for key '{key}', applying legacy pattern matching"
                )

                # Legacy pattern matching fallback (kept for backward compatibility)
                # TO be switched off
                if key in ("Date", "Time"):
                    pass  # non-physical metadata columns — no unit needed
                elif key == "timestamp":
                    self.units[key] = ("time", None)
                elif key == "t":
                    self.units[key] = ("t", ureg.second)
                elif key == "Field":
                    self.units[key] = ("B", ureg.tesla)
                elif key.startswith("I"):
                    self.units[key] = ("I", ureg.ampere)
                elif key.startswith("U"):
                    self.units[key] = ("U", ureg.volt)
                elif key.startswith("T") or key == "teb" or key == "tsb":
                    self.units[key] = ("T", ureg.degC)
                elif key.startswith("Rpm"):
                    self.units[key] = ("Rpm", ureg.rpm)
                elif key.startswith("DR"):
                    self.units[key] = ("%", ureg.percent)
                elif key.startswith("Flo"):
                    self.units[key] = ("Q", ureg.liter / ureg.second)
                elif key == "debitbrut":
                    self.units[key] = ("Q", ureg.meter**3 / ureg.hour)
                elif key.startswith("HP") or key.startswith("BP"):
                    self.units[key] = ("P", ureg.bar)
                elif key == "Pmagnet" or key == "Ptot" or key.startswith("Power"):
                    self.units[key] = ("Power", ureg.megawatt)
                elif key == "Q":
                    self.units[key] = ("Preac", ureg.megavar)
                else:
                    logger.warning(f"Units: no unit defined for key '{key}' — skipping")

        if debug:
            logger.debug(f"Units: {self.Keys}")
            for key, values in self.units.items():
                symbol = values[0]
                unit = values[1]
                logger.debug(f"{key}: symbol={symbol}, unit={unit:~P}")

    def getUnitKey(self, key: str) -> tuple:
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
        return self.units[key]

    # --- timestamp validation ----------------------------------------

    def _validate_start_timestamp(self) -> None:
        """Cross-check ``start_timestamp`` against the first ``Date``/``Time`` data row.

        Called automatically at the end of ``__init__``, before any cleanup.

        * If ``Date`` and ``Time`` columns are present, parse the first row into a
          :class:`~pandas.Timestamp`.
        * When ``start_timestamp`` was not derived from the filename (``None``), set it
          from the data.
        * When the filename-derived value disagrees with the data, emit a warning and
          overwrite with the authoritative data value.
        """
        if "Date" not in self.Keys or "Time" not in self.Keys:
            return
        assert isinstance(self.Data, pd.DataFrame)
        if self.Data.empty:
            return
        try:
            date_str = str(self.Data["Date"].iloc[0])
            time_str = str(self.Data["Time"].iloc[0])
            data_ts = pd.Timestamp(
                datetime.strptime(f"{date_str} {time_str}", "%Y.%m.%d %H:%M:%S")
            )
        except (ValueError, TypeError):
            logger.warning(
                f"_validate_start_timestamp: cannot parse Date/Time from first row of {self.FileName!r}"
            )
            return

        if self.start_timestamp is None:
            logger.debug(
                f"_validate_start_timestamp: {self.FileName!r} — start_timestamp set from data: {data_ts}"
            )
            self.start_timestamp = data_ts
        elif self.start_timestamp != data_ts:
            logger.warning(
                f"_validate_start_timestamp: {self.FileName!r} — filename timestamp {self.start_timestamp} "
                f"differs from data timestamp {data_ts}; using data value -- aka {data_ts}"
            )
            self.start_timestamp = data_ts

    # --- cleanup / reshape -------------------------------------------

    def cleanupData(  # noqa: N802
        self,
        keys_to_remove: list[str] | None = None,
        keys_to_rename: dict[str, str] | None = None,
        keys_to_add: dict[str, str] | None = None,
        debug: bool = False,
    ) -> int:
        self._ensure_data_loaded()
        logger.debug(f"Clean up Data: filename={self.FileName}, keys={self.Keys}")
        assert isinstance(self.Data, pd.DataFrame)

        if keys_to_add:
            logger.debug(f"cleanupData: adding keys {list(keys_to_add.keys())}")
            existing_keys = [key for key in keys_to_add if key in self.Keys]
            if existing_keys:
                logger.warning(
                    f"cleanupData: keys {existing_keys} already exist in DataFrame, skipping addition"
                )
            for key, formula in keys_to_add.items():
                self.addData(key, formula, debug=debug)

        if keys_to_rename:
            logger.debug(f"cleanupData: renaming keys {keys_to_rename}")
            missing_keys = [key for key in keys_to_rename if key not in self.Keys]
            if missing_keys:
                logger.warning(
                    f"cleanupData: keys {missing_keys} not found in DataFrame, cannot rename"
                )
            target_exists = [
                new_key for new_key in keys_to_rename.values() if new_key in self.Keys
            ]
            if target_exists:
                logger.warning(
                    f"cleanupData: target keys {target_exists} already exist in DataFrame, will be overwritten"
                )
            self.renameData(keys_to_rename)

        if keys_to_remove:
            logger.debug(f"cleanupData: removing keys {keys_to_remove}")
            missing_keys = [key for key in keys_to_remove if key not in self.Keys]
            if missing_keys:
                logger.warning(
                    f"cleanupData: keys {missing_keys} not found in DataFrame, cannot remove"
                )
            self.removeData(keys_to_remove)

        self.Keys = _dataframe_keys(self.Data)

        if "t" in self.Keys:
            from .utils.duplicates import find_duplicates

            self.Data = find_duplicates(self.Data, self.FileName, "t")
            self.Keys = _dataframe_keys(self.Data)

        import re

        Fkeys = set(
            [_key for _key in self.Keys if re.match(r"Flow\w+", _key)]
            + [_key for _key in self.Keys if re.match(r"Rpm\w+", _key)]
            + [_key for _key in self.Keys if re.match(r"HP\w+", _key)]
            + [_key for _key in self.Keys if re.match(r"\w+_ref", _key)]
            + [_key for _key in self.Keys if re.match(r"Pmagnet", _key)]
            + [_key for _key in self.Keys if re.match(r"Ptot", _key)]
            + [_key for _key in self.Keys if re.match(r"Idcct\d", _key)]
            + [
                _key
                for _key in self.Keys
                if re.match(r"(Supra)?Field|TotalField", _key)
            ]
            + [_key for _key in self.Keys if re.match(r"TAlimout", _key)]
        )

        logger.debug(
            f"zero columns: {natsorted(self.Data.columns[(self.Data == 0).all()].values.tolist())}",
        )

        empty_cols: list = [
            col
            for col in self.Data.columns[(self.Data == 0).all()].values.tolist()
            if col not in Fkeys
        ]
        logger.info(f"empty cols (to drop): {natsorted(empty_cols)}")
        if empty_cols:
            self.Data = self.Data.drop(empty_cols, axis=1)
            self.Keys = _dataframe_keys(self.Data)

        dropped_columns = _get_duplicate_columns(self.Data)
        really_dropped_columns = natsorted(
            [
                col
                for col in dropped_columns
                if not col.startswith("Ucoil") and col not in Fkeys
            ]
        )
        logger.info(
            f"duplicate columns (others than Ucoil* and Fkeys): {natsorted(really_dropped_columns)}"
        )
        if really_dropped_columns:
            self.Data = self.Data.drop(really_dropped_columns, axis=1)
            self.Keys = _dataframe_keys(self.Data)

        return 0

    def removeData(self, keys: list) -> int:  # noqa: N802
        assert isinstance(self.Data, pd.DataFrame)
        for key in keys:
            if key in self.Keys:
                del self.Data[key]
            else:
                logger.warning(
                    f"removeData: cannot remove '{key}', key not found - skipping"
                )
        self.Keys = _dataframe_keys(self.Data)
        return 0

    def renameData(self, columns: dict) -> None:  # noqa: N802
        assert isinstance(self.Data, pd.DataFrame)
        missing = [old for old in columns if old not in self.Keys]
        if missing:
            logger.warning(
                f"renameData: keys {missing} not found in DataFrame, skipping"
            )
            columns = {old: new for old, new in columns.items() if old not in missing}
        if not columns:
            return
        # A target name that already exists (and is not itself being renamed away)
        # would silently overwrite that column — raise instead.
        source_keys = set(columns.keys())
        conflicts = {
            old: new
            for old, new in columns.items()
            if new in self.Keys and new not in source_keys
        }
        if conflicts:
            raise ValueError(
                f"renameData: target name(s) already exist in DataFrame and would "
                f"be silently overwritten: {conflicts}"
            )
        self.Data.rename(columns=columns, inplace=True)
        self.Keys = _dataframe_keys(self.Data)

    # --- compute / add -----------------------------------------------

    def addData(  # noqa: N802
        self,
        key: str,
        formula: str,
        unit: str | tuple | None = None,
        debug: bool = False,
        label: str = "",
        description: str = "",
    ) -> int:
        from pint.errors import UndefinedUnitError

        from .magnetdata_base import FieldMeta, _make_ureg

        assert isinstance(self.Data, pd.DataFrame)
        if key in self.Keys:
            logger.warning(
                f"addData: key '{key}' already exists in DataFrame, skipping addition"
            )
        else:
            self.Data.eval(formula, inplace=True)
            self.Keys = _dataframe_keys(self.Data)
            if isinstance(unit, tuple) and len(unit) == 2:
                symbol, pint_unit = unit
                self.units[key] = (symbol, pint_unit)
                self.field_meta[key] = FieldMeta(
                    symbol=symbol, unit=pint_unit, label=label, description=description
                )
            elif isinstance(unit, str) and unit:
                try:
                    ureg = _make_ureg()
                    parsed = ureg.parse_expression(unit)
                    pint_unit = parsed.units if hasattr(parsed, "units") else parsed
                    self.units[key] = (key, pint_unit)
                    self.field_meta[key] = FieldMeta(
                        symbol=key, unit=pint_unit, label=label, description=description
                    )
                except (ValueError, UndefinedUnitError):
                    self.Units(debug)
            else:
                self.Units(debug)
        return 0

    def computeData(  # noqa: N802
        self,
        method: Any,
        key: str,
        kparams: list,
        unit: tuple | str | None = None,
        debug: bool = False,
        label: str = "",
        description: str = "",
    ) -> None:
        from pint.errors import UndefinedUnitError

        from .magnetdata_base import FieldMeta, _make_ureg

        logger.debug(f"computeData: Key={key}")
        if key in self.Keys:
            logger.warning(f"Key {key} already exists in DataFrame")
            return
        assert isinstance(self.Data, pd.DataFrame)
        data = []
        for values in self.Data[kparams].values.tolist():
            data.append(method(*values))
        self.Data[key] = data
        self.Keys = _dataframe_keys(self.Data)
        if isinstance(unit, tuple) and len(unit) == 2:
            symbol, pint_unit = unit
            self.units[key] = (symbol, pint_unit)
            self.field_meta[key] = FieldMeta(
                symbol=symbol, unit=pint_unit, label=label, description=description
            )
        elif isinstance(unit, str) and unit:
            try:
                ureg = _make_ureg()
                parsed = ureg.parse_expression(unit)
                pint_unit = parsed.units if hasattr(parsed, "units") else parsed
                self.units[key] = (key, pint_unit)
                self.field_meta[key] = FieldMeta(
                    symbol=key, unit=pint_unit, label=label, description=description
                )
            except (ValueError, UndefinedUnitError):
                self.Units(debug)
        else:
            self.Units(debug)
        logger.debug("done")

    # --- time utilities ----------------------------------------------

    def getStartDate(self, group: str | None = None) -> tuple:  # noqa: N802
        res: tuple = ()
        if "Date" in self.Keys and "Time" in self.Keys:
            start_date = self.Data["Date"].iloc[0]  # type: ignore[index]
            start_time = self.Data["Time"].iloc[0]  # type: ignore[index]
            end_date = self.Data["Date"].iloc[-1]  # type: ignore[index]
            end_time = self.Data["Time"].iloc[-1]  # type: ignore[index]
            res = (start_date, start_time, end_date, end_time)
        return res

    def getDuration(self, group: str | None = None) -> float:  # noqa: N802
        if "t" in self.Keys:
            assert isinstance(self.Data, pd.DataFrame)
            return float(self.Data["t"].iloc[-1] - self.Data["t"].iloc[0])  # type: ignore[index]
        logger.warning("magnetdata.getDuration: no t key")
        logger.warning(f"available keys are: {self.Keys}")
        return 0.0

    def addTime(self, time_zone: str = "Europe/Paris") -> int:  # noqa: N802
        """Compute ``t`` (elapsed seconds) and ``timestamp`` (naive UTC) columns.

        Drops ``Date`` and ``Time`` after conversion.  The ``timestamp`` column
        stores naive UTC regardless of the local timezone of the source data.
        Call this before :meth:`extractTimeData` or any ``timestamp``-based plot.

        :param time_zone: IANA timezone of the source ``Date``/``Time`` columns
            (default ``"Europe/Paris"``).
        """
        self._ensure_data_loaded()
        assert isinstance(self.Data, pd.DataFrame)
        if "Date" not in self.Keys or "Time" not in self.Keys:
            raise RuntimeError(
                f"MagnetData/AddTime {self.FileName}: cannot add t[s] columnn: no Date or Time columns"
            )

        try:
            self.Data["Date"] = pd.to_datetime(
                self.Data.Date, cache=True, format="%Y.%m.%d"
            )
        except (ValueError, TypeError):
            raise RuntimeError(
                f"MagnetData/AddTime {self.FileName}: failed to convert Date"
            ) from None

        try:
            self.Data["Time"] = pd.to_timedelta(self.Data.Time)
        except (ValueError, TypeError):
            raise RuntimeError(
                f"MagnetData/AddTime {self.FileName}: failed to convert Time"
            ) from None

        try:
            _local_ts = self.Data.Date + self.Data.Time
        except (ValueError, TypeError):
            raise RuntimeError(
                f"MagnetData/AddTime {self.FileName}: failed to create timestamp column"
            ) from None

        self.Data["_timestamp"] = _local_ts
        from .utils.duplicates import find_duplicates

        self.Data = find_duplicates(self.Data, self.FileName, "_timestamp")

        t0 = self.Data["_timestamp"].iloc[0]
        self.Data["t"] = (self.Data["_timestamp"] - t0).dt.total_seconds()

        # Convert local → naive UTC
        self.Data["timestamp"] = series_local_to_utc_naive(
            self.Data["_timestamp"], time_zone
        )

        self.Data.drop(["Date", "Time", "_timestamp"], axis=1, inplace=True)
        self.Keys = _dataframe_keys(self.Data)
        return 0

    def shiftTime(self, dt: float) -> int:  # noqa: N802
        if "t" in self.Keys:
            self.Data["t"] = self.Data["t"] + dt  # type: ignore[index]
        else:
            raise RuntimeError(
                f"MagnetData/shiftTime {self.FileName}: cannot shift t[s] columnn: no t column"
            )
        return 0

    def get_time_range(self) -> tuple:
        """Return ``(start_timestamp, end_timestamp)`` for the dataset.

        ``start_timestamp`` comes from the filename (set at construction time).
        ``end_timestamp`` is derived as ``start_timestamp + getDuration()``.

        Falls back to parsing the first/last Date+Time data rows when
        ``start_timestamp`` could not be extracted from the filename.
        """
        if self.start_timestamp is not None:
            duration = self.getDuration()
            self.end_timestamp = self.start_timestamp + pd.Timedelta(seconds=duration)
            return (self.start_timestamp, self.end_timestamp)

        # Fallback: derive both timestamps from the Date/Time data columns
        if "Date" not in self.Keys or "Time" not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.get_time_range: no Date/Time columns in {self.FileName}"
            )
        assert isinstance(self.Data, pd.DataFrame)
        tformat = "%Y.%m.%d %H:%M:%S"
        start_str = f"{self.Data['Date'].iloc[0]} {self.Data['Time'].iloc[0]}"
        end_str = f"{self.Data['Date'].iloc[-1]} {self.Data['Time'].iloc[-1]}"
        self.start_timestamp = pd.Timestamp(datetime.strptime(start_str, tformat))
        self.end_timestamp = pd.Timestamp(datetime.strptime(end_str, tformat))
        return (self.start_timestamp, self.end_timestamp)

    # --- extract -----------------------------------------------------

    def extractData(self, keys: list[str]) -> pd.DataFrame:  # noqa: N802
        logger.debug(f"extractData: filename={self.FileName}, keys={keys}")
        for key in keys:
            if key not in self.Keys:
                raise RuntimeError(
                    f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {key} key"
                )
        logger.debug("extractData: Done")
        return pd.concat([self.Data[key] for key in keys], axis=1)  # type: ignore[index]

    def extractDataThreshold(
        self, key: str, threshold: float
    ) -> pd.DataFrame:  # noqa: N802
        assert isinstance(self.Data, pd.DataFrame)
        if key not in self.Keys:
            raise RuntimeError(
                f"extractData: key={key} - no such keys in dataframe (valid keys are: {self.Keys}"
            )
        return self.Data.loc[self.Data[key] >= threshold]

    def extractTimeData(  # noqa: N802
        self, timerange: str, group: str | None = None, time_zone: str = "Europe/Paris"
    ) -> pd.DataFrame:
        """Return rows whose ``timestamp`` falls within *timerange*.

        :param timerange: ``"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"`` in local
            time (the ``time_zone`` timezone).  Both boundaries are inclusive.
        :param group: unused for pandas data; accepted for interface compatibility.
        :param time_zone: IANA timezone of the datetime strings in *timerange*
            (default ``"Europe/Paris"``).
        :raises RuntimeError: if :meth:`addTime` has not been called yet.
        """
        assert isinstance(self.Data, pd.DataFrame)
        if "timestamp" not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.extractTimeData: call addTime() before extractTimeData()"
            )
        logger.debug(f"Select data from {timerange}")
        t_start, t_end = timerange_to_utc(timerange, time_zone)
        return self.Data[
            self.Data["timestamp"].between(t_start, t_end, inclusive="both")
        ]

    # --- persist / display -------------------------------------------

    def saveData(self, keys: list[str], filename: str) -> int:  # noqa: N802
        assert isinstance(self.Data, pd.DataFrame)
        self.Data[keys].to_csv(filename, sep="\t", index=False, header=True)
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

        assert   isinstance(self.Data, pd.DataFrame)
        df: pd.DataFrame = self.Data.copy()

        # Convert UTC timestamp → naive local time for display
        if x == "timestamp":
            df["timestamp"] = series_utc_to_local_naive(df["timestamp"], time_zone)

        kwargs: dict = {"x": x, "y": y, "ax": ax, "alpha": alpha, "grid": False}
        if color is not None:
            kwargs["color"] = color
        if marker is not None:
            kwargs["marker"] = marker
        if linestyle is not None:
            kwargs["linestyle"] = linestyle
        if markevery is not None:
            kwargs["markevery"] = markevery

        if normalize:
            ymax = abs(df[y].max())
            df[y] /= ymax
            kwargs["label"] = f"{label or y} (norm with {ymax:.3e} {yunit:~P})"
        elif label is not None:
            kwargs["label"] = label

        df.plot(**kwargs)

        if yunit is not None:
            logger.info(
                f"ysymbol={ysymbol}, yunit={yunit:~P}, labeling y-axis accordingly"
            )
            plt.ylabel(f"{ysymbol} [{yunit:~P}]")

        (xsymbol, xunit) = self.getUnitKey(x)
        if xunit is not None:
            logger.info(
                f"plotData: xsymbol={xsymbol}, xunit={xunit:~P}, labeling x-axis accordingly"
            )
            plt.xlabel(f"{xsymbol} [{xunit:~P}]")

    def stats(self, key: str | None = None) -> pd.DataFrame | None:
        from tabulate import tabulate

        logger.info("magnetdata.stats")
        assert isinstance(self.Data, pd.DataFrame)
        if key is not None:
            if key in self.Keys:
                logger.info(
                    tabulate(self.Data[key].describe(), headers="keys", tablefmt="psql")
                )
            else:
                raise RuntimeError(
                    f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {key} key"
                )
        else:
            df = self.Data.describe(include="all")
            print(tabulate(df.values.tolist(), headers=list(df.columns), tablefmt="psql"))
        return None

    def info(self) -> None:
        from tabulate import tabulate

        print(f"magnetdata: {self.FileName}, Type={self.Type.name}")

        # Optionally load descriptions from the defs file
        field_defs: dict = {}
        if self.defs_file is not None:
            try:
                from .field_defs import load_defs

                field_defs = load_defs(self.defs_file)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(
                    f"info: cannot load field definitions from {self.defs_file!r}: {exc}"
                )

        from natsort import natsorted

        rows = []
        for key in natsorted(self.Keys):
            description = field_defs.get(key, {}).get("description", "")
            if key in self.units:
                symbol, unit = self.units[key]
                unit_str = f"{unit:~P}" if unit is not None else ""
            else:
                symbol = ""
                unit_str = ""
            rows.append([key, description, symbol, unit_str])

        print(
            tabulate(
                rows, headers=["Key", "Description", "Symbol", "Unit"], tablefmt="psql"
            )
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Type={self.Type!r}, Groups={self.Groups!r}, Keys={self.Keys!r}, Data={self.Data!r})"

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def fromtxt(
        cls, name: str, defs_file: str | None = "pupitre-defs.json"
    ) -> PandasMagnetData:
        """Create from a pupitre .txt file.

        Only the first data row is read at construction time so that
        ``_validate_start_timestamp`` can cross-check the filename timestamp.
        The full file is loaded lazily on the first call to
        :meth:`_ensure_data_loaded` (triggered by :meth:`addTime`,
        :meth:`cleanupData`, or :meth:`getPandasData`).
        """
        from .utils.validation import FileFormatError, validate_txt_format

        if os.path.splitext(name)[-1] != ".txt":
            raise FileFormatError(f"{name}: expected .txt extension")
        validate_txt_format(name)
        _csv_kwargs = {"sep": r"\s+", "engine": "python", "skiprows": 1}
        with open(name) as f:
            stub = pd.read_csv(f, **_csv_kwargs, nrows=1)
        Keys = _dataframe_keys(stub)
        return cls(name, {}, Keys, stub, defs_file=defs_file, _read_kwargs=_csv_kwargs)

    @classmethod
    def fromcsv(cls, name: str, defs_file: str | None = None) -> PandasMagnetData:
        """Create from a CSV file."""
        from .utils.validation import validate_csv_format

        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=0)
            Keys = _dataframe_keys(Data)
        return cls(name, {}, Keys, Data, defs_file=defs_file)

    @classmethod
    def fromStringIO(  # noqa: N802
        cls,
        name: str,
        sep: str = r"\s+",
        skiprows: int = 1,
        defs_file: str | None = None,
    ) -> PandasMagnetData:
        """Create from a StringIO / in-memory string."""
        from io import StringIO

        Data = pd.DataFrame()
        Keys: list[str] = []
        try:
            Data = pd.read_csv(
                StringIO(name), sep=sep, engine="python", skiprows=skiprows
            )
            Keys = _dataframe_keys(Data)
        except (pd.errors.ParserError, ValueError, OSError):
            logger.error("fromStringIO: trouble loading data")
            with open("wrongdata.txt", "w", newline="\n") as fo:
                fo.write(name)
        return cls("stringIO", {}, Keys, Data, defs_file=defs_file)


# ---------------------------------------------------------------------------
# Thin subclasses — differ only in Type and/or loading logic
# ---------------------------------------------------------------------------


class EnsightMagnetData(PandasMagnetData):
    """Ensight CSV-backed data (Type=2).

    Identical to :class:`PandasMagnetData` except ``Type == 2``.
    The existing bug where ``getData`` raised ``RuntimeError`` for Type=2
    is fixed here by simply inheriting the working pandas implementation.
    """

    _TYPE: DataType = DataType.ENSIGHT

    @classmethod
    def fromensight(cls, name: str, defs_file: str | None = None) -> EnsightMagnetData:
        """Create from a CSV ensight file."""
        from .utils.validation import validate_file_exists

        validate_file_exists(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=2)
            Keys = _dataframe_keys(Data)
        return cls(name, {}, Keys, Data, defs_file=defs_file)


class BProfileMagnetData(PandasMagnetData):
    """B-profile CSV data (Index, Position, Profile columns, Type=0)."""

    _TYPE: DataType = DataType.PUPITRE

    @classmethod
    def frombprofile(
        cls, name: str, defs_file: str | None = None
    ) -> BProfileMagnetData:
        """Create from a bprofile CSV file (Index, Position, Profile columns)."""
        from .utils.validation import validate_csv_format

        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=0)
            Keys = _dataframe_keys(Data)
        return cls(name, {}, Keys, Data, defs_file=defs_file)


class FeelppMagnetData(PandasMagnetData):
    """feelpp simulation CSV data (Type=0)."""

    _TYPE: DataType = DataType.PUPITRE

    @classmethod
    def fromfeelpp(
        cls, name: str, skiprows: int = 0, defs_file: str | None = None
    ) -> FeelppMagnetData:
        """Create from a feelpp simulation CSV file."""
        from .utils.validation import validate_csv_format

        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=skiprows)
            Keys = _dataframe_keys(Data)
        return cls(name, {}, Keys, Data, defs_file=defs_file)
