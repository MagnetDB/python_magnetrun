"""TdmsMagnetData — TDMS-backed magnet data (pigbrother files)."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any

import pandas as pd

from .magnetdata_base import MagnetDataBase

logger = logging.getLogger(__name__)


class TdmsMagnetData(MagnetDataBase):
    """TDMS-backed magnet data.

    ``self.Data`` is a ``dict[str, pd.DataFrame]`` keyed by group name.
    ``self.Type`` is ``1``.
    """

    @property
    def Type(self) -> int:  # type: ignore[override]
        return 1

    # --- core data access --------------------------------------------

    def getTdmsData(self, group: str, channel: str | list[str] | None) -> pd.DataFrame:
        if not isinstance(self.Data, dict):
            raise Exception(
                f"MagnetData/getTdmsData: {self.FileName} - expect Data to be a dict"
            )
        if channel is None or not channel:
            return self.Data[group]
        return self.Data[group][channel]

    def getData(self, key: list[str] | str | None = None) -> pd.DataFrame:
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
                f"magnetata:getData for tdms - expect only one group - got {len(groups)}"
            )

        return self.getTdmsData(groups[0], channels)

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
            "Champ_magn": ("B", ureg.gauss),
        }

        for entry in _pig_units:
            if entry in key:
                return _pig_units[entry]

        return ()

    def Units(self, debug: bool = False) -> None:  # noqa: N802
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

        for entry in self.Data:
            if entry == "t":
                self.units["t"] = ("t", ureg.second)
            else:
                group = entry
                if "/" in entry:
                    (group, channel) = entry.split("/")
                    if channel == "t":
                        self.units[entry] = ("t", ureg.second)
                self.units[entry] = self.PigBrotherUnits(group)

        if debug:
            logger.debug(f"Units: {self.Keys}")

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
        (group, channel) = key.split("/")
        return self.PigBrotherUnits(group)

    def renameData(self, columns: dict) -> None:  # noqa: N802
        """TDMS data does not support renaming channels; this is a no-op."""

    # --- compute / add -----------------------------------------------

    def addData(  # noqa: N802
        self, key: str, formula: str, unit: str | None = None, debug: bool = False
    ) -> int:
        (group, channel) = key.split("/")
        logger.debug(f"add: key={key} - group={group}, channel={channel}")

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
            self.Groups[group][channel] = {
                "wf_increment": self.Groups[group][first_key]["wf_increment"]
            }

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
        unit: tuple | None = None,
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
        if group is None:
            return 0.0
        channel = list(self.Groups[group].keys())[0]
        ordered_dict = self.Groups[group][channel]
        dt = ordered_dict["wf_increment"]
        samples = ordered_dict["wf_samples"]
        return float(dt * samples)

    def addTdmsTime(self, group: str | None = None) -> int:  # noqa: N802
        """Add a ``'t'`` column to group(s) in Data.

        Uses ``wf_increment`` and ``wf_start_offset`` from the first channel's
        properties to compute ``t = index * dt + t_offset``.
        """
        assert isinstance(self.Data, dict)

        if group is not None and group not in self.Data:
            raise RuntimeError(
                f"MagnetData/addTdmsTime {self.FileName}: group '{group}' not found in Data"
            )

        groups_to_process = [group] if group is not None else list(self.Data.keys())

        for gname in groups_to_process:
            if gname == "Infos":
                continue
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
                    "wf_increment": dt,
                    "wf_start_offset": t_offset,
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
                if "t" not in self.Data[group].columns:  # type: ignore[index]
                    self.addTdmsTime(group=group)
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
        return self.Data[group][channel].loc[self.Data[group][channel] >= threshold]  # type: ignore[index]

    def extractTimeData(  # noqa: N802
        self, timerange: str, group: str | None = None
    ) -> pd.DataFrame:
        trange = timerange.split(";")
        logger.debug(f"Select data from {trange[0]} to {trange[1]}")
        return self.Data[group]["timestamp"].between(trange[0], trange[1], inclusive="both")  # type: ignore[index]

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

        if xchannel == "t" and "t" not in self.Data[xgroup].columns:  # type: ignore[index]
            self.addTdmsTime(group=xgroup)
        df = self.Data[xgroup].copy()  # type: ignore[index]

        if normalize:
            ymax = abs(df[ychannel].max())
            df[ychannel] /= ymax
            df.plot(
                x=xchannel,
                y=ychannel,
                ax=ax,
                alpha=alpha,
                label=f"{ychannel} (norm with {ymax:.3e} {yunit:~P})",
                grid=True,
            )
            del df
        else:
            df.plot(x=xchannel, y=ychannel, alpha=alpha, ax=ax, grid=True)

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
            for group in self.Data:
                logger.info(f"stats[{group}]: ")
                df = self.Data[group].describe(include="all")  # type: ignore[index]
                logger.info(tabulate(df, headers="keys", tablefmt="psql"))
        return None

    def info(self) -> None:
        from collections import OrderedDict

        from tabulate import tabulate

        logger.info(f"magnetdata: {self.FileName}, Type={self.Type}")
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
