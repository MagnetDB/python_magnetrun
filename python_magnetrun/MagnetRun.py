"""Main module."""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from .magnetdata import load_magnetdata
from .magnetdata_base import DataType, MagnetDataBase
from .magnetdata_pandas import PandasMagnetData
from .runetl import prepareData
from .utils.downsampling import DownsampleConfig
from .utils.timezone import ensure_utc_naive, local_to_utc_naive

logger = logging.getLogger(__name__)


def load_mrun(
    filename: str,
    housing: str = "unknown",
    site: str = "",
    time_zone: str = "Europe/Paris",
    auto_resolve: bool = True,
    **kwargs,
) -> "MagnetRun":
    """Load a MagnetRun from a file, dispatching on extension.

    - ``.tdms`` → :meth:`MagnetRun.fromtdms`
    - ``.txt``  → :meth:`MagnetRun.fromtxt`
    - ``.csv``  → :meth:`MagnetRun.fromcsv`

    :param filename: path to the data file (can be a simple filename or full path)
    :param housing: housing name (default ``"unknown"``)
    :param site: site name
    :param time_zone: local timezone for timestamp conversion (txt/tdms)
    :param auto_resolve: when True, automatically search in standard data directories
        if the file is not found at the given path. Uses the same resolution logic
        as the CLI (searches current dir, then extension-specific data dirs, then
        data_dir/housing). Default: True.
    :param kwargs: extra keyword arguments forwarded to the underlying classmethod
    :raises ValueError: if the file extension is not recognised
    :raises FileNotFoundError: if the file cannot be found
    """
    import os

    resolved_filename = filename

    # If file doesn't exist and auto_resolve is enabled, try standard data directories
    if auto_resolve and not os.path.exists(filename):
        from .data_dirs import PIGBROTHER_DATA_DIR, PUPITRE_DATA_DIR
        from .utils.files import expand_input_files

        logger.debug(
            f"load_mrun: file '{filename}' not found, attempting auto-resolution"
        )

        datadir = {
            ".tdms": PIGBROTHER_DATA_DIR,
            ".txt": PUPITRE_DATA_DIR,
        }

        # Use housing from filename if not provided
        resolve_housing = housing if housing and housing != "unknown" else None

        resolved = expand_input_files([filename], datadir, housing=resolve_housing)

        if not resolved:
            raise FileNotFoundError(
                f"load_mrun: file not found: {filename}\n"
                f"  Searched: current directory, {datadir.get(os.path.splitext(filename)[-1], 'N/A')}"
                f"{f', {datadir.get(os.path.splitext(filename)[-1])}/{resolve_housing}' if resolve_housing else ''}"
            )

        resolved_filename = resolved[0]
        logger.info(f"load_mrun: resolved '{filename}' → '{resolved_filename}'")

    ext = os.path.splitext(resolved_filename)[-1].lower()
    if ext == ".tdms":
        return MagnetRun.fromtdms(housing, site, resolved_filename, time_zone=time_zone)
    elif ext == ".txt":
        return MagnetRun.fromtxt(
            housing, site, resolved_filename, time_zone=time_zone, **kwargs
        )
    elif ext == ".csv":
        return MagnetRun.fromcsv(housing, site, resolved_filename)
    else:
        raise ValueError(f"load_mrun: unsupported file extension {ext!r}")


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
        data: MagnetDataBase | None = None,
        start_time: datetime | None = None,
    ):
        """default constructor"""
        self.Housing = housing
        self.Site = site
        self.MagnetData = data
        self.StartTime = start_time

    @classmethod
    def fromtdms(
        cls,
        housing: str,
        site: str,
        filename: str,
        time_zone: str = "Europe/Paris",
    ) -> "MagnetRun":
        """create from a tdms file"""
        # print(f"MagnetRun:fromtdms: {filename}", flush=True)
        # with open(filename, "r") as f:
        logger.debug(
            f"MagnetRun:fromtdms: housing={housing}, site={site}, filename={filename}"
        )
        data = load_magnetdata(filename)
        prepareData(data, housing)
        data.Units()

        group = list(data.Groups.keys())[0]
        channel = list(data.Groups[group].keys())[0]
        # Use wf_start_time only — do NOT add wf_start_offset.
        # addTdmsTime computes t = index*dt + wf_start_offset, so wf_start_offset
        # is already encoded in the t values.  Adding it to StartTime would
        # cause double-counting when computing delta_t for multi-file alignment.
        ts = pd.Timestamp(data.Groups[group][channel]["wf_start_time"])
        wf_start_offset = data.Groups[group][channel].get("wf_start_offset", 0.0)
        # wf_start_time is UTC; normalise to naive UTC regardless of tzinfo.
        start_time = ensure_utc_naive(ts)
        logger.debug(
            f"magnetrun.fromtdms: start_time={start_time} (naive UTC), "
            f"wf_start_offset={wf_start_offset} s (already in t values, not added to StartTime)"
        )
        return cls(housing, site, data, start_time=start_time)

    @classmethod
    def fromtxt(
        cls,
        housing: str,
        site: str,
        filename: str,
        keys_to_remove: list[str] | None = None,
        keys_to_rename: dict[str, str] | None = None,
        keys_to_add: dict[str, str] | None = None,
        time_zone: str = "Europe/Paris",
    ) -> "MagnetRun":
        """create from a txt file"""
        logger.debug(
            f"MagnetRun/fromtxt: housing={housing}, site={site}, filename={filename}"
        )
        # with open(filename, "r") as f:
        # insert = f.readline().split()[-1]
        data = load_magnetdata(filename)
        # logger.debug(f"MagnetRun/from_txt: data={data}")
        res = data.getStartDate()
        logger.debug(f"MagnetRun.from_txt: getStartDate={res}")
        prepareData(data, housing)
        logger.debug(
            f"MagnetRun/from_txt: after prepareData - data keys={data.getKeys()}"
        )
        data.Units()
        (start_date, start_time, end_date, end_time) = res

        # Combine start_date (YYYY.MM.DD) and start_time (HH:MM:SS) into datetime.
        # The timestamp from pupitre data is local time; convert to naive UTC so it
        # is directly comparable with timestamps from pigbrother (.tdms) files.
        start_t = datetime.strptime(f"{start_date} {start_time}", "%Y.%m.%d %H:%M:%S")
        start_t = local_to_utc_naive(start_t, time_zone)
        logger.debug(f"MagnetRun/from_txt: start_t={start_t} (naive UTC)")
        return cls(housing, site, data, start_time=start_t)

    @classmethod
    def fromcsv(cls, housing: str, site: str, filename: str) -> "MagnetRun":
        """create from a csv file"""
        data = load_magnetdata(filename)
        return cls(housing, site, data)

    @classmethod
    def fromStringIO(cls, housing: str, site: str, name: str) -> "MagnetRun":
        """create from a stringIO"""
        logger.info(f"MagnetRun/fromStringIO: housing={housing}, site={site}")
        from io import StringIO

        insert = "Unknown"
        data = PandasMagnetData(filename="", Groups={}, Keys=[])
        try:
            ioname = StringIO(name)
            # TODO rework: get item 2 otherwise set to unknown
            headers = ioname.readline().split()
            if len(headers) >= 2:
                insert = headers[1]
            if not site.startswith(insert):
                logger.debug(f"MagnetRun:fromStringIO: site={site}, insert={insert}")
            data = PandasMagnetData.fromStringIO(name)
            logger.debug(
                f"MagnetRun:fromStringIO: data keys({len(data.getKeys())}): {data.getKeys()}"
            )
            prepareData(data, housing)

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

    def getDomain(self) -> str:
        return "operational"

    def get_time_range(self):
        """Delegate to the underlying MagnetData object."""
        return self.MagnetData.get_time_range()

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

    def getMData(self) -> MagnetDataBase:
        """return Magnet Data object"""
        if self.MagnetData is not None:
            return self.MagnetData
        else:
            raise RuntimeError("no magnetdata attached to this magnetrun")

    def getData(
        self,
        key: str = "",
        downsample: DownsampleConfig | None = None,
    ) -> Any:
        """return Data, optionally downsampled"""
        if self.MagnetData is not None:
            return self.MagnetData.getData(key, downsample=downsample)
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

    def getDataFrame(
        self,
        downsample: DownsampleConfig | None = None,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Return data as DataFrame(s), optionally downsampled.

        For pupitre data returns a single DataFrame.
        For pigbrother/TDMS data returns a list of DataFrames, one per group.
        """
        if self.MagnetData is None:
            raise RuntimeError("MagnetRun.getDataFrame: no MagnetData associated")
        if self.MagnetData.Type == DataType.PUPITRE:
            return self.MagnetData.getData(downsample=downsample)
        elif self.MagnetData.Type == DataType.TDMS:
            return [
                self.MagnetData.getData(group, downsample=downsample)
                for group in self.MagnetData.Groups
            ]
        else:
            raise RuntimeError(
                f"MagnetRun.getDataFrame: unsupported type {self.MagnetData.Type}"
            )

    def saveData(self, filename: str) -> None:
        """save Data to file"""
        if self.MagnetData is not None:
            self.MagnetData.saveData(self.MagnetData.getKeys(), filename)
