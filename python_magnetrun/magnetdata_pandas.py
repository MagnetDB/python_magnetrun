"""PandasMagnetData and thin pandas-backed subclasses."""

from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from natsort import natsorted

from .magnetdata_base import MagnetDataBase

logger = logging.getLogger(__name__)


class PandasMagnetData(MagnetDataBase):
    """Pandas-backed magnet data (pupitre .txt, .csv, StringIO).

    ``self.Data`` is always a :class:`pandas.DataFrame`.
    ``self.Type`` is ``0`` unless overridden by a subclass.
    """

    _TYPE: int = 0  # overridden by EnsightMagnetData → 2

    def __init__(
        self,
        filename: str,
        Groups: dict,
        Keys: list[str],
        Data: pd.DataFrame | None = None,
    ) -> None:
        super().__init__(filename, Groups, Keys, Data)

    # --- abstract property -------------------------------------------

    @property
    def Type(self) -> int:  # type: ignore[override]
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

    def Units(self, debug: bool = False) -> None:  # noqa: N802
        """Populate ``self.units`` from column names."""
        from pint import UnitRegistry
        from pint.errors import UndefinedUnitError

        ureg: UnitRegistry = UnitRegistry()
        for defn, unit in [
            ("percent = 1 / 100 = %", "percent"),
            ("ppm = 1e-6 = ppm", "ppm"),
            ("var = 1", "var"),
        ]:
            try:
                ureg.parse_units(unit)
            except UndefinedUnitError:
                ureg.define(defn)

        for key in self.Keys:
            if key == "timestamp":
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

    def cleanupData_legacy(self) -> int:  # noqa: N802
        """Remove empty/duplicate columns (legacy implementation)."""
        warnings.warn(
            "prepareData_legacy is deprecated and will be removed in a future version. "
            "Use prepareData instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.debug(
            f"Clean up Data (legacy): filename={self.FileName}, keys={self.Keys}"
        )
        assert isinstance(self.Data, pd.DataFrame)
        import re

        init_Ikeys = natsorted(
            [_key for _key in self.Keys if re.match(r"Icoil\d+", _key)]
        )
        logger.debug(f"init_Ikeys: {init_Ikeys}")
        Fkeys = [_key for _key in self.Keys if re.match(r"Flow\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Rpm\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"HP\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"\w+_ref", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Pmagnet", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Ptot", _key)]
        Fkeys += [
            "teb",
            "tsb",
            "debitbrut",
        ]

        def getDuplicateColumns(df: pd.DataFrame) -> list[str]:
            duplicateColumnNames: set[str] = set()
            for x in range(df.shape[1]):
                col = df.iloc[:, x]
                for y in range(x + 1, df.shape[1]):
                    otherCol = df.iloc[:, y]
                    if col.equals(otherCol):
                        duplicateColumnNames.add(df.columns.values[y])
            return list(duplicateColumnNames)

        logger.debug(
            f"zero columns: {natsorted(self.Data.columns[(self.Data == 0).all()].values.tolist())}",
        )

        empty_cols = [
            col
            for col in self.Data.columns[(self.Data == 0).all()].values.tolist()
            if not col.startswith("Flow") and not col.startswith("Field")
        ]
        _empty_Ikeys = natsorted(
            [_key for _key in empty_cols if re.match(r"Icoil\d+", _key)]
        )
        _df = self.Data
        if empty_cols:
            _df = self.Data.drop(empty_cols, axis=1)

        dropped_columns = getDuplicateColumns(_df)
        really_dropped_columns = natsorted(
            [col for col in dropped_columns if not col.startswith("Ucoil")]
        )
        _df.drop(really_dropped_columns, axis=1, inplace=True)

        Ukeys = natsorted(
            [
                str(_key)
                for _key in _df.columns.values.tolist()
                if re.match(r"Ucoil\d+", _key)
            ]
        )

        from itertools import groupby

        Uindex = [int(ukey.replace("Ucoil", "")) for ukey in Ukeys]
        gb = groupby(enumerate(Uindex), key=lambda x: x[0] - x[1])
        all_groups = ([i[1] for i in g] for _, g in gb)
        Uprobes = list(filter(lambda x: len(x) > 1, all_groups))
        if not Uprobes:
            raise RuntimeError(f"{self.FileName}: CleanUpData no Uprobes found")

        UH = [f"Ucoil{i}" for i in Uprobes[0]]
        _df["UH"] = _df[UH].sum(axis=1)
        logger.debug(f"UH: {UH}")
        if len(Uprobes) > 1:
            UB = [f"Ucoil{i}" for i in Uprobes[1]]
            _df["UB"] = _df[UB].sum(axis=1)
            logger.debug(f"UB: {UB}")

        Ikeys = natsorted(
            [
                _key
                for _key in _df.columns.values.tolist()
                if re.match(r"Icoil\d+", _key)
            ]
        )
        logger.debug(f"IKeys = {Ikeys} ({len(Ikeys)})")
        if Ikeys:
            if len(Ikeys) == 1:
                logger.debug(
                    f"{self.FileName}: check if {init_Ikeys[-1]} or {init_Ikeys[-2]} in _df"
                )
                if init_Ikeys[-1] not in Ikeys and init_Ikeys[-2] not in Ikeys:
                    _df = pd.concat([_df, self.Data[init_Ikeys[-2]]], axis=1)
                else:
                    _df = pd.concat([_df, self.Data[init_Ikeys[0]]], axis=1)

                Ikeys = natsorted(
                    [
                        _key
                        for _key in _df.columns.values.tolist()
                        if re.match(r"Icoil\d+", _key)
                    ]
                )

            elif len(Ikeys) == 2:
                logger.debug("need to check consistancy")

            else:
                logger.debug(
                    f"{self.FileName}:try to cure dataset - got {Ikeys} expect at most 2 values",
                )
                ikeys = self.Data[Ikeys]
                remove_Ikeys = []
                for i in range(len(Ikeys)):
                    for j in range(i + 1, len(Ikeys)):
                        diff = ikeys[Ikeys[i]] - ikeys[Ikeys[j]]
                        error = diff.mean()
                        stderror = diff.std()
                        logger.debug(
                            f"diff[{Ikeys[i]}_{Ikeys[j]}: mean={error}, std={stderror}"
                        )
                        if abs(error) <= 1.0e-2:
                            remove_Ikeys.append(Ikeys[j])

                logger.debug(f"remove_Ikeys: {remove_Ikeys}")
                if remove_Ikeys:
                    _df.drop(remove_Ikeys, axis=1, inplace=True)

                Ikeys = natsorted(
                    [
                        _key
                        for _key in _df.columns.values.tolist()
                        if re.match(r"Icoil\d+", _key)
                    ]
                )

                if len(Ikeys) == 1:
                    logger.debug(
                        f"{self.FileName}: check if {init_Ikeys[-1]} or {init_Ikeys[-2]} in _df"
                    )
                    if (
                        init_Ikeys[-1] not in _df.columns.values.tolist()
                        and init_Ikeys[-2] not in _df.columns.values.tolist()
                    ):
                        _df = pd.concat([_df, self.Data[init_Ikeys[-2]]], axis=1)
                    else:
                        _df = pd.concat([_df, self.Data[init_Ikeys[0]]], axis=1)

                elif len(Ikeys) > 2:
                    _df[Ikeys].to_csv(f"{self.FileName}.ikey")
                    raise RuntimeError(
                        f"{self.FileName}: strange number of Ikeys detected - got {Ikeys} expect at most 2 values"
                    )

        else:
            Ukeys = natsorted(
                [
                    str(_key)
                    for _key in _df.columns.values.tolist()
                    if re.match(r"Ucoil\d+", _key)
                ]
            )
            for i, key in enumerate(Ukeys):
                Ukeys[i] = key.replace("U", "I")
            Ikeys = [Ukeys[0], Ukeys[-1]]
            _df = pd.concat([_df, self.Data[Ikeys[0]], self.Data[Ikeys[1]]], axis=1)

        _df_keys = _df.columns.values.tolist()
        for key in Fkeys:
            if key not in _df_keys:
                _df = pd.concat([_df, self.Data[key]], axis=1)

        self.Data = _df
        self.Keys = self.Data.columns.values.tolist()
        logger.debug(f"cleanupData_legacy: final keys --> self.Keys = {self.Keys}")
        return 0

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

        import re

        Fkeys = [_key for _key in self.Keys if re.match(r"Flow\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Rpm\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"HP\w+", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"\w+_ref", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Pmagnet", _key)]
        Fkeys += [_key for _key in self.Keys if re.match(r"Ptot", _key)]

        def getDuplicateColumns(df: pd.DataFrame) -> list[str]:
            duplicateColumnNames: set[str] = set()
            for x in range(df.shape[1]):
                col = df.iloc[:, x]
                for y in range(x + 1, df.shape[1]):
                    otherCol = df.iloc[:, y]
                    if col.equals(otherCol):
                        duplicateColumnNames.add(df.columns.values[y])
            return list(duplicateColumnNames)

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

        dropped_columns = getDuplicateColumns(_df)
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
            logger.info(tabulate(df, headers="keys", tablefmt="psql"))
        return None

    def info(self) -> None:
        logger.info(f"magnetdata: {self.FileName}, Type={self.Type}")
        logger.info("keys:")
        for key in self.Keys:
            logger.info(f"\t{key}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Type={self.Type!r}, Groups={self.Groups!r}, Keys={self.Keys!r}, Data={self.Data!r})"


# ---------------------------------------------------------------------------
# Thin subclasses — differ only in Type and/or loading logic
# ---------------------------------------------------------------------------


class EnsightMagnetData(PandasMagnetData):
    """Ensight CSV-backed data (Type=2).

    Identical to :class:`PandasMagnetData` except ``Type == 2``.
    The existing bug where ``getData`` raised ``RuntimeError`` for Type=2
    is fixed here by simply inheriting the working pandas implementation.
    """

    _TYPE: int = 2


class BProfileMagnetData(PandasMagnetData):
    """B-profile CSV data (Index, Position, Profile columns, Type=0)."""

    _TYPE: int = 0


class FeelppMagnetData(PandasMagnetData):
    """feelpp simulation CSV data (Type=0)."""

    _TYPE: int = 0
