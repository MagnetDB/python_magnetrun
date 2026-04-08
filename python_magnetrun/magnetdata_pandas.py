"""PandasMagnetData and thin pandas-backed subclasses."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from natsort import natsorted

from .magnetdata_base import DataType, MagnetDataBase

logger = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(filename, Groups, Keys, Data, defs_file)

    # --- abstract property -------------------------------------------

    @property
    def Type(self) -> DataType:
        return self._TYPE

    # --- core data access --------------------------------------------

    def getPandasData(self, key: list[str] | str | None) -> pd.DataFrame:
        """Return Data or a selection by *key*."""
        if key is None:
            if not isinstance(self.Data, pd.DataFrame):
                raise Exception(
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
                raise Exception(
                    f"MagnetData/Data({key}): {self.FileName}: cannot get data for key={item}: no such key"
                )
        return self.Data[selected_keys]  # type: ignore[index]

    def getData(self, key: list[str] | str | None = None) -> pd.DataFrame:
        return self.getPandasData(key)

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

    # --- cleanup / reshape -------------------------------------------

    def cleanupData(  # noqa: N802
        self,
        keys_to_remove: list[str] | None = None,
        keys_to_rename: dict[str, str] | None = None,
        keys_to_add: dict[str, str] | None = None,
        debug: bool = False,
    ) -> int:
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

        self.Keys = self.Data.columns.values.tolist()

        # Add legacy stuff for backward compatibility - to be removed in a future version
        import re

        Fkeys = [_key for _key in self.Keys if re.match(r"Flow\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Rpm\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"HP\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"\w+_ref", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Pmagnet", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Ptot", _key)]

        logger.debug(
            f"zero columns: {natsorted(self.Data.columns[(self.Data == 0).all()].values.tolist())}",
        )

        empty_cols = [
            col
            for col in self.Data.columns[(self.Data == 0).all()].values.tolist()
            if not col.startswith("Flow") and not col.startswith("Field")
        ]
        _empty_Ikeys = natsorted([_key for _key in empty_cols])
        logger.info(f"empty cols: {natsorted(empty_cols)}")

        _df = self.Data
        if empty_cols:
            _df = self.Data.drop(empty_cols, axis=1)

        dropped_columns = _get_duplicate_columns(_df)
        really_dropped_columns = natsorted(
            [col for col in dropped_columns if not col.startswith("Ucoil")]
        )
        logger.info(
            f"duplicate columns (others than Ucoil*): {natsorted(really_dropped_columns)}"
        )

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
        self.Keys = self.Data.columns.values.tolist()
        return 0

    def renameData(self, columns: dict) -> None:  # noqa: N802
        assert isinstance(self.Data, pd.DataFrame)
        existing_renames = {
            old: new for old, new in columns.items() if old in self.Keys
        }
        if len(existing_renames) < len(columns):
            missing = [old for old in columns if old not in self.Keys]
            logger.warning(f"renameData: keys {missing} not found, skipping")
        # A target name that already exists (and is not itself being renamed away)
        # would silently overwrite that column — raise instead.
        source_keys = set(existing_renames.keys())
        conflicts = {
            old: new
            for old, new in existing_renames.items()
            if new in self.Keys and new not in source_keys
        }
        if conflicts:
            raise ValueError(
                f"renameData: target name(s) already exist in DataFrame and would "
                f"be silently overwritten: {conflicts}"
            )
        if existing_renames:
            self.Data.rename(columns=existing_renames, inplace=True)
        self.Keys = self.Data.columns.values.tolist()

    # --- compute / add -----------------------------------------------

    def addData(  # noqa: N802
        self, key: str, formula: str, unit: str | None = None, debug: bool = False
    ) -> int:
        assert isinstance(self.Data, pd.DataFrame)
        if key in self.Keys:
            logger.warning(
                f"addData: key '{key}' already exists in DataFrame, skipping addition"
            )
        else:
            self.Data.eval(formula, inplace=True)
            self.Keys = self.Data.columns.values.tolist()
            if unit:
                self.units[key] = unit
            else:
                self.Units(debug)
        return 0

    def computeData(  # noqa: N802
        self,
        method: Any,
        key: str,
        kparams: list,
        unit: tuple | None = None,
        debug: bool = False,
    ) -> None:
        logger.debug(f"computeData: Key={key}")
        if key in self.Keys:
            logger.warning(f"Key {key} already exists in DataFrame")
            return
        assert isinstance(self.Data, pd.DataFrame)
        data = []
        for values in self.Data[kparams].values.tolist():
            data.append(method(*values))
        self.Data[key] = data
        self.Keys = self.Data.columns.values.tolist()
        if unit:
            self.units[key] = unit
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
        duration = 0.0
        if "timestamp" in self.Keys:
            start_time = self.Data["timestamp"].iloc[0]  # type: ignore[index]
            end_time = self.Data["timestamp"].iloc[-1]  # type: ignore[index]
            dt = end_time - start_time
            duration = float(dt.seconds)
        else:
            logger.warning("magnetdata.getDuration: no timestamp key")
            logger.warning(f"available keys are: {self.Keys}")
        return duration

    def addTime(self) -> int:  # noqa: N802
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
            self.Data["timestamp"] = self.Data.Date + self.Data.Time
        except (ValueError, TypeError):
            raise RuntimeError(
                f"MagnetData/AddTime {self.FileName}: failed to create timestamp column"
            ) from None
        else:
            t0 = self.Data.iloc[0]["timestamp"]

        from .utils.duplicates import find_duplicates

        self.Data = find_duplicates(self.Data, self.FileName, "timestamp")

        self.Data["t"] = self.Data.apply(
            lambda row: (row.timestamp - t0).total_seconds(),
            axis=1,
        )

        times = self.Data["t"].to_numpy()
        _dt = np.diff(times)

        self.Data.drop(["Date", "Time"], axis=1, inplace=True)
        self.Keys = self.Data.columns.values.tolist()
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
        """Return ``(start_datetime, end_datetime)`` parsed from Date+Time columns."""
        if "Date" not in self.Keys or "Time" not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.get_time_range: no Date/Time columns in {self.FileName}"
            )
        assert isinstance(self.Data, pd.DataFrame)
        tformat = "%Y.%m.%d %H:%M:%S"
        start_str = f"{self.Data['Date'].iloc[0]} {self.Data['Time'].iloc[0]}"
        end_str = f"{self.Data['Date'].iloc[-1]} {self.Data['Time'].iloc[-1]}"
        start_dt = datetime.strptime(start_str, tformat)
        end_dt = datetime.strptime(end_str, tformat)
        return (start_dt, end_dt)

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
        self, timerange: str, group: str | None = None
    ) -> pd.DataFrame:
        assert isinstance(self.Data, pd.DataFrame)
        trange = timerange.split(";")
        logger.debug(f"Select data from {trange[0]} to {trange[1]}")
        return self.Data[
            self.Data["Time"].between(trange[0], trange[1], inclusive="both")
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
    ) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams["text.usetex"] = True

        if x != "t" and x != "timestamp" and x not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no x={x} key (valid keys= {self.Keys})"
            )

        if y not in self.Keys:
            raise Exception(
                f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}: no {y} key (valid keys: {self.Keys})"
            )

        assert isinstance(self.Data, pd.DataFrame)
        (ysymbol, yunit) = self.getUnitKey(y)
        if normalize:
            df = self.Data.copy()
            ymax = abs(df[y].max())
            df[y] /= ymax
            df.plot(
                x=x,
                y=y,
                ax=ax,
                alpha=alpha,
                label=f"{y} (norm with {ymax:.3e} {yunit:~P})",
                grid=True,
            )
            del df
        else:
            self.Data.plot(x=x, y=y, ax=ax, alpha=alpha, grid=True)

        if yunit is not None:
            plt.ylabel(f"{ysymbol} [{yunit:~P}]")

        (xsymbol, xunit) = self.getUnitKey(x)
        if xunit is not None:
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
            print(tabulate(df, headers="keys", tablefmt="psql"))
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

        rows = []
        for key in self.Keys:
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
    def fromtxt(cls, name: str, defs_file: str | None = "pupitre-defs.json") -> PandasMagnetData:
        """Create from a pupitre .txt file."""
        from .utils.validation import validate_txt_format
        validate_txt_format(name)
        with open(name) as f:
            if os.path.splitext(name)[-1] != ".txt":
                raise RuntimeError(f"fromtxt: expect a txt filename - got {name}")
            Data = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)
            Keys = Data.columns.values.tolist()
        return cls(name, {}, Keys, Data, defs_file=defs_file)

    @classmethod
    def fromcsv(cls, name: str, defs_file: str | None = None) -> PandasMagnetData:
        """Create from a CSV file."""
        from .utils.validation import validate_csv_format
        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=0)
            Keys = Data.columns.values.tolist()
        return cls(name, {}, Keys, Data, defs_file=defs_file)

    @classmethod
    def fromStringIO(  # noqa: N802
        cls, name: str, sep: str = r"\s+", skiprows: int = 1, defs_file: str | None = None
    ) -> PandasMagnetData:
        """Create from a StringIO / in-memory string."""
        from io import StringIO

        Data = pd.DataFrame()
        Keys: list[str] = []
        try:
            Data = pd.read_csv(StringIO(name), sep=sep, engine="python", skiprows=skiprows)
            Keys = Data.columns.values.tolist()
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
            Keys = Data.columns.values.tolist()
        return cls(name, {}, Keys, Data, defs_file=defs_file)


class BProfileMagnetData(PandasMagnetData):
    """B-profile CSV data (Index, Position, Profile columns, Type=0)."""

    _TYPE: DataType = DataType.PUPITRE

    @classmethod
    def frombprofile(cls, name: str, defs_file: str | None = None) -> BProfileMagnetData:
        """Create from a bprofile CSV file (Index, Position, Profile columns)."""
        from .utils.validation import validate_csv_format
        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=0)
            Keys = Data.columns.values.tolist()
        return cls(name, {}, Keys, Data, defs_file=defs_file)


class FeelppMagnetData(PandasMagnetData):
    """feelpp simulation CSV data (Type=0)."""

    _TYPE: DataType = DataType.PUPITRE

    @classmethod
    def fromfeelpp(cls, name: str, skiprows: int = 0, defs_file: str | None = None) -> FeelppMagnetData:
        """Create from a feelpp simulation CSV file."""
        from .utils.validation import validate_csv_format
        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=skiprows)
            Keys = Data.columns.values.tolist()
        return cls(name, {}, Keys, Data, defs_file=defs_file)
