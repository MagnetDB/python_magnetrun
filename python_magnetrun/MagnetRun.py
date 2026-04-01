"""Main module."""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from .magnetdata import MagnetData
from .runetl import prepareData_legacy

logger = logging.getLogger(__name__)


class MagnetRun:
    """
    Magnet Run

    Housing: name of the housing
    Site: name of site (magnetdb sense)
    MagnetData: pandas dataframe
    """

    def __init__(
        self,
        housing: str = "unknown",
        site: str = "",
        data: MagnetData | None = None,
        start_time: datetime | None = None,
    ):
        """default constructor"""
        self.Housing = housing
        self.Site = site
        self.MagnetData = data
        self.StartTime = start_time

    @classmethod
    def fromtdms(cls, site: str, insert: str, filename: str) -> "MagnetRun":
        """create from a tdms file"""
        # print(f"MagnetRun:fromtdms: {filename}", flush=True)
        # with open(filename, "r") as f:
        data = MagnetData.fromtdms(filename)
        data.Units()

        group = list(data.Groups.keys())[0]
        channel = list(data.Groups[group].keys())[0]
        start_t = pd.Timestamp(data.Groups[group][channel]["wf_start_time"])
        offset_t = pd.Timedelta(data.Groups[group][channel]["wf_start_offset"])
        start_time = (start_t + offset_t).to_pydatetime()
        logger.debug(
            f"magnetrun.fromtdms: start_time={start_time}, type={type(start_time)}"
        )
        return cls(site, insert, data, start_time=start_time)

    @classmethod
    def fromtxt(
        cls,
        housing: str,
        site: str,
        filename: str,
        keys_to_remove: list[str] | None = None,
        keys_to_rename: dict[str, str] | None = None,
        keys_to_add: dict[str, str] | None = None,
    ) -> "MagnetRun":
        """create from a txt file"""
        logger.debug(
            f"MagnetRun/fromtxt: housing={housing}, site={site}, filename={filename}"
        )
        # with open(filename, "r") as f:
        # insert = f.readline().split()[-1]
        data = MagnetData.fromtxt(filename)
        logger.debug(f"data: {data}")
        res = data.getStartDate()
        print("loading raw pupitre data:", data.getKeys())
        prepareData_legacy(data, housing)
        print("after prepareData_legacy: data keys=", data.getKeys())
        # TODO: switch to prepareData once legacy is removed:
        # prepareData(data, housing, keys_to_remove=keys_to_remove,
        #             keys_to_rename=keys_to_rename, keys_to_add=keys_to_add)
        logger.debug(f"res: {res}")
        data.Units()
        (start_date, start_time, end_date, end_time) = res

        # print("magnetrun.fromtxt: data=", data)
        # Combine start_date (YYYY.MM.DD) and start_time (HH:MM:SS) into datetime
        start_t = datetime.strptime(f"{start_date} {start_time}", "%Y.%m.%d %H:%M:%S")
        logger.debug(f"magnetrun.fromtxt: start_t={start_t}, type={type(start_t)}")
        return cls(housing, site, data, start_time=start_t)

    @classmethod
    def fromcsv(
        cls, housing: str, site: str, filename: str
    ) -> "MagnetRun":
        """create from a csv file"""
        data = MagnetData.fromcsv(filename)
        return cls(housing, site, data)

    @classmethod
    def fromStringIO(
        cls, housing: str, site: str, name: str
    ) -> "MagnetRun":
        """create from a stringIO"""
        # print(f'MagnetRun/fromStringIO: housing={housing}, site={site}')
        from io import StringIO

        insert = "Unknown"
        data = MagnetData(filename="", Groups={}, Keys=[])
        try:
            ioname = StringIO(name)
            # TODO rework: get item 2 otherwise set to unknown
            headers = ioname.readline().split()
            if len(headers) >= 2:
                insert = headers[1]
            if not site.startswith(insert):
                logger.debug(f"MagnetRun:fromStringIO: site={site}, insert={insert}")
            data = MagnetData.fromStringIO(name)
            # print(f'data keys({len(data.getKeys())}): {data.getKeys()}')
            prepareData_legacy(data, housing)
            # print(f'prepareData: data keys({len(data.getKeys())}): {data.getKeys()}')

        except (ValueError, KeyError, AttributeError, IndexError) as e:
            logger.error(
                f"cannot load data for {housing}, {insert} insert, {site} site: {e}"
            )
            raise RuntimeError(
                f"cannot load data for {housing}, {insert} insert, {site} site"
            ) from e

        data.Units()
        return cls(housing, site, data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Housing={self.Housing!r}, Site={self.Site!r}, MagnetData={self.MagnetData!r})"

    def getInsert(self) -> str:
        """returns Insert"""
        import os

        filename = self.MagnetData.FileName  # type: ignore[union-attr]
        f_extension = os.path.splitext(filename)[-1]

        return filename.replace(f_extension, "")

    def getSite(self) -> str:
        """returns Site"""
        return self.Site

    def getHousing(self) -> str:
        """returns Housing"""
        return self.Housing

    def setSite(self, site: str) -> None:
        """set Site"""
        self.Site = site

    def setHousing(self, housing: str) -> None:
        """set Housing"""
        self.Housing = housing

    def getType(self) -> int:
        """returns Data Type"""
        if self.MagnetData is not None:
            return self.MagnetData.Type
        else:
            raise RuntimeError("MagnetRun.getType: no MagnetData associated")

    def getMData(self) -> MagnetData:
        """return Magnet Data object"""
        if self.MagnetData is not None:
            return self.MagnetData
        else:
            raise RuntimeError("no magnetdata attached to this magnetrun")

    def getData(self, key: str = "") -> Any:
        """return Data"""
        if self.MagnetData is not None:
            return self.MagnetData.getData(key)
        else:
            raise RuntimeError("MagnetRun.getData: no MagnetData associated")

    def getUnit(self, key: str = "") -> tuple:
        """return Unit"""
        if self.MagnetData is not None:
            return self.MagnetData.getUnitKey(key)
        else:
            raise RuntimeError("MagnetRun.getData: no MagnetData associated")

    def getKeys(self) -> list[str]:
        """return list of Data keys"""
        if self.MagnetData is not None:
            return self.MagnetData.Keys
        else:
            raise RuntimeError("MagnetRun.getKeys: no MagnetData associated")

    def getStats(self, field: str | None = None) -> pd.DataFrame | None:
        """return basic stats"""
        if self.MagnetData is not None:
            return self.MagnetData.stats(field)
        else:
            raise RuntimeError("MagnetRun.getStats: no MagnetData associated")

    def saveData(self, filename: str) -> None:
        """save Data to file"""
        if self.MagnetData is not None:
            if isinstance(self.MagnetData.Data, pd.DataFrame):
                self.MagnetData.Data.to_csv(
                    filename, sep="\t", index=False, header=True
                )
            else:
                raise RuntimeError(
                    f"MagnetRun.save: unsupported type of Data ({type(self.MagnetData.Data)})"  # noqa: E501
                )
