"""Main module."""

import logging
import re
import pandas as pd
from natsort import natsorted
from datetime import datetime

logger = logging.getLogger(__name__)

from .magnetdata import MagnetData


def prepareData_legacy(data: MagnetData, housing: str, debug: bool = False):
    """Prepare magnet run data by adding computed fields and renaming columns (LEGACY VERSION).

    Adds IH_ref/IB_ref computed currents and renames Flow/Rpm/Tin/HP columns
    with H/B suffixes appropriate for the given housing configuration.

    :param data: MagnetData object to prepare in-place
    :type data: MagnetData
    :param housing: Housing name (e.g. "M8", "M9", "M10")
    :type housing: str
    :param debug: Enable debug output, defaults to False
    :type debug: bool, optional
    """
    # get start/end
    (start_date, start_time, end_date, end_time) = data.getStartDate()
    logger.debug(
        f"prepareData_legacy: start_date={start_date}, start_time={start_time}, end_date={end_date}, end_time={end_time}"
    )

    # add timestamp
    data.addTime()
    # print(f'addTime done')

    # get duration
    duration = data.getDuration()
    # print(f'duration={duration}')

    # TODO use a dict struct to simplify this?
    # shall check if key exist beforehand
    if housing == "M9":
        # print('M9 housing case')
        data.addData("IH_ref", "IH_ref = Idcct1 + Idcct2")
        data.addData("IB_ref", "IB_ref = Idcct3 + Idcct4")

        # FlowH = Flow1, FlowB = Flow2
        for field in ["Flow", "Rpm", "Tin", "HP"]:
            data.renameData(
                columns={f"{field}1": f"{field}H", f"{field}2": f"{field}B"}
            )

    elif housing in ["M8", "M10"]:
        # print('M8/M10 housing case')
        data.addData("IH_ref", "IH_ref = Idcct3 + Idcct4")
        data.addData("IB_ref", "IB_ref = Idcct1 + Idcct2")

        # FlowH = Flow2, FlowB = Flow1
        for field in ["Flow", "Rpm", "Tin", "HP"]:
            data.renameData(
                columns={f"{field}1": f"{field}B", f"{field}2": f"{field}H"}
            )
    # what about M1, M5 and M7???

    # data.removeData(["Idcct1", "Idcct2", "Idcct3", "Idcct4"])

    data.cleanupData_legacy(debug=debug)
    Ikey = natsorted([_key for _key in data.getKeys() if re.match(r"Icoil\d+", _key)])
    logger.debug(f"MagnetRun/prepareData_legacy: housing={housing}, Ikey={Ikey}")

    data.renameData(columns={f"{Ikey[0]}": "IH"})
    data.renameData(columns={f"{Ikey[-1]}": "IB"})

    logger.debug(f"MagnetRun.prepareData_legacy: data.keys={data.getKeys()}")


def prepareData(
    data: MagnetData,
    housing: str,
    keys_to_remove: list[str] | None = None,
    keys_to_rename: dict[str, str] | None = None,
    keys_to_add: dict[str, str] | None = None,
    debug: bool = False,
):
    """Prepare magnet run data by adding computed fields and renaming columns.

    This method adds timestamp and performs cleanup with flexible configuration.
    Housing-specific operations (IH_ref/IB_ref, Flow/Rpm/Tin/HP renaming, Icoil→IH/IB)
    should now be specified via the keys_to_add and keys_to_rename parameters.
    
    All custom operations are handled by the cleanupData() method.

    :param data: MagnetData object to prepare in-place
    :type data: MagnetData
    :param housing: Housing name (e.g. "M8", "M9", "M10") - for reference/logging
    :type housing: str
    :param keys_to_remove: list of column names to remove, defaults to None
    :type keys_to_remove: list[str] | None, optional
    :param keys_to_rename: dict mapping old column names to new names, defaults to None
    :type keys_to_rename: dict[str, str] | None, optional
    :param keys_to_add: dict mapping new column names to their formulas, defaults to None
    :type keys_to_add: dict[str, str] | None, optional
    :param debug: Enable debug output, defaults to False
    :type debug: bool, optional
    """
    # get start/end
    (start_date, start_time, end_date, end_time) = data.getStartDate()
    logger.debug(
        f"prepareData: start_date={start_date}, start_time={start_time}, end_date={end_date}, end_time={end_time}"
    )

    # add timestamp
    data.addTime()

    # get duration
    duration = data.getDuration()

    # NOTE: All custom operations (keys_to_add, keys_to_rename, keys_to_remove) 
    # are now handled by cleanupData() method

    # Standard housing-specific operations
    # NOTE: Housing-specific operations should now be handled via keys_to_add, keys_to_rename parameters
    # TODO use a dict struct to simplify this?
    # shall check if key exist beforehand
    # if housing == "M9":
    #     data.addData("IH_ref", "IH_ref = Idcct1 + Idcct2")
    #     data.addData("IB_ref", "IB_ref = Idcct3 + Idcct4")
    #
    #     # FlowH = Flow1, FlowB = Flow2
    #     for field in ["Flow", "Rpm", "Tin", "HP"]:
    #         data.renameData(
    #             columns={f"{field}1": f"{field}H", f"{field}2": f"{field}B"}
    #         )
    #
    # elif housing in ["M8", "M10"]:
    #     data.addData("IH_ref", "IH_ref = Idcct3 + Idcct4")
    #     data.addData("IB_ref", "IB_ref = Idcct1 + Idcct2")
    #
    #     # FlowH = Flow2, FlowB = Flow1
    #     for field in ["Flow", "Rpm", "Tin", "HP"]:
    #         data.renameData(
    #             columns={f"{field}1": f"{field}B", f"{field}2": f"{field}H"}
    #         )
    # what about M1, M5 and M7???

    # Perform cleanup with flexible parameters
    data.cleanupData(
        keys_to_remove=keys_to_remove,
        keys_to_rename=keys_to_rename,
        keys_to_add=keys_to_add,
        debug=debug
    )
    
    # Rename Icoil columns to IH/IB
    # NOTE: This should now be handled via keys_to_rename parameter
    # Ikey = natsorted([_key for _key in data.getKeys() if re.match(r"Icoil\d+", _key)])
    # logger.debug(f"MagnetRun/prepareData: housing={housing}, Ikey={Ikey}")
    #
    # data.renameData(columns={f"{Ikey[0]}": "IH"})
    # data.renameData(columns={f"{Ikey[-1]}": "IB"})

    logger.debug(f"MagnetRun.prepareData: data.keys={data.getKeys()}")


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
    def fromtdms(cls, site, insert, filename):
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
        logger.debug(f"magnetrun.fromtdms: start_time={start_time}, type={type(start_time)}")
        return cls(site, insert, data, start_time=start_time)

    @classmethod
    def fromtxt(cls, housing: str, site: str, filename: str, debug: bool = False):
        """create from a txt file"""
        logger.debug(f"MagnetRun/fromtxt: housing={housing}, site={site}, filename={filename}")
        # with open(filename, "r") as f:
        # insert = f.readline().split()[-1]
        data = MagnetData.fromtxt(filename)
        logger.debug(f"data: {data}")
        res = data.getStartDate()
        prepareData(data, housing, debug=debug)
        logger.debug(f"res: {res}")
        data.Units()
        (start_date, start_time, end_date, end_time) = res

        # print("magnetrun.fromtxt: data=", data)
        # Combine start_date (YYYY.MM.DD) and start_time (HH:MM:SS) into datetime
        start_t = datetime.strptime(f"{start_date} {start_time}", "%Y.%m.%d %H:%M:%S")
        logger.debug(f"magnetrun.fromtxt: start_t={start_t}, type={type(start_t)}")
        return cls(housing, site, data, start_time=start_t)

    @classmethod
    def fromcsv(cls, housing: str, site: str, filename: str, debug: bool = False):
        """create from a csv file"""
        data = MagnetData.fromcsv(filename)
        return cls(housing, site, data)

    @classmethod
    def fromStringIO(cls, housing: str, site: str, name: str, debug: bool = False):
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
            prepareData(data, housing, debug=debug)
            # print(f'prepareData: data keys({len(data.getKeys())}): {data.getKeys()}')

        except Exception as e:
            logger.error(f"cannot load data for {housing}, {insert} insert, {site} site: {e}")
            raise RuntimeError(
                f"cannot load data for {housing}, {insert} insert, {site} site"
            ) from e

        data.Units()
        return cls(housing, site, data)

    def __repr__(self):
        return "%s(Housing=%r, Site=%r, MagnetData=%r)" % (
            self.__class__.__name__,
            self.Housing,
            self.Site,
            self.MagnetData,
        )

    def getInsert(self):
        """returns Insert"""
        import os

        filename = self.MagnetData.FileName
        f_extension = os.path.splitext(filename)[-1]

        return filename.replace(f_extension, "")

    def getSite(self):
        """returns Site"""
        return self.Site

    def getHousing(self):
        """returns Housing"""
        return self.Housing

    def setSite(self, site):
        """set Site"""
        self.Site = site

    def getType(self):
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

    def getData(self, key: str = ""):
        """return Data"""
        if self.MagnetData is not None:
            return self.MagnetData.getData(key)
        else:
            raise RuntimeError("MagnetRun.getData: no MagnetData associated")

    def getUnit(self, key: str = ""):
        """return Unit"""
        if self.MagnetData is not None:
            return self.MagnetData.getUnitKey(key)
        else:
            raise RuntimeError("MagnetRun.getData: no MagnetData associated")

    def getKeys(self):
        """return list of Data keys"""
        if self.MagnetData is not None:
            return self.MagnetData.Keys
        else:
            raise RuntimeError("MagnetRun.getKeys: no MagnetData associated")

    def getStats(self, field: str = None):
        """return basic stats"""
        if self.MagnetData is not None:
            return self.MagnetData.stats(field)
        else:
            raise RuntimeError("MagnetRun.getStats: no MagnetData associated")

    def saveData(self, filename: str):
        """save Data to file"""
        if self.MagnetData is not None:
            if isinstance(self.MagnetData.Data, pd.DataFrame):
                self.MagnetData.Data.to_csv(
                    filename, sep=str("\t"), index=False, header=True
                )
            else:
                raise RuntimeError(
                    f"MagnetRun.save: unsupported type of Data ({type(self.MagnetData.Data)})"
                )
