#!/usr/bin/env python3

"""Magnet Record Object"""

import datetime
import json
import logging
import warnings
from typing import Any

from .connect import download

logger = logging.getLogger(__name__)


class MRecord:
    """
    timestamp
    housing
    assembly
    link
    """

    def __init__(self, timestamp: datetime.datetime, housing: str, assembly: str, link: str) -> None:
        """default constructor"""
        self.timestamp = timestamp
        self.housing = housing
        self.assembly = assembly
        self.link = link

    def __repr__(self) -> str:
        """
        representation of object
        """
        return f"{self.__class__.__name__}(timestamp={self.timestamp!r}, housing={self.housing!r}, assembly={self.assembly!r}, link={self.link!r})"

    def getTimestamp(self) -> datetime.datetime:
        """get timestamp"""
        return self.timestamp

    def getHousing(self) -> str:
        """get experimental site"""
        return self.housing

    def getAssembly(self) -> str:
        """get experimental assembly magnet"""
        return self.assembly

    def getSite(self) -> str:  # noqa: N802
        warnings.warn(
            "getSite() is deprecated, use getAssembly() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.getAssembly()

    def getLink(self) -> str:
        """get link"""
        return self.link

    def setTimestamp(self, timestamp: datetime.datetime) -> None:
        """set timestamp"""
        self.timestamp = timestamp

    def setAssembly(self, assembly: str) -> None:
        """set Assembly"""
        self.assembly = assembly

    def setSite(self, site: str) -> None:  # noqa: N802
        warnings.warn(
            "setSite() is deprecated, use setAssembly() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.setAssembly(site)

    def setLink(self, link: str) -> None:
        """set Link"""
        self.link = link

    def getData(self, session: Any, url: str) -> str:
        """download record"""
        if not session:
            raise Exception("MRecord.download: no session defined")

        logger.debug(f"Downloading data for record {self} from {url} with link {self.link}")
        params = {"file": self.link, "download": "1"}
        data = download(session, url, params, self.link)
        return data

    def getDataFilename(self) -> str:
        filename = self.link.replace("../../../", "")
        filename = filename.replace("/", "_").replace("%20", "-")
        return filename

    def saveData(self, data: str, datadir: str = ".") -> None:
        filename = self.getDataFilename()
        if datadir != ".":
            filename = f"{datadir}/{filename}"
        logger.debug(f"Saving data to {filename}")
        with open(filename, "w", newline="\n") as fo:
            fo.write(data)

    def to_json(self) -> str:
        """
        convert to json
        """
        from . import deserialize

        return json.dumps(self, default=deserialize.serialize_instance, sort_keys=True, indent=4)

    def __eq__(self, other: object) -> bool:
        """compare MRecords"""
        if isinstance(other, MRecord):
            return (
                self.timestamp == other.timestamp
                and self.assembly == other.assembly
                and self.link == other.link
            )
        return False

    # def __le__(self, other):
    #     """compare MRecords"""
    #     if (isinstance(other, MRecord)):
    #         return self.timestamp <= other.timestamp:
    #     return False

    # def __ge__(self, other):
    #     """compare MRecords"""
    #     if (isinstance(other, MRecord)):
    #         return self.timestamp >= other.timestamp:
    #     return False
