"""MagnetDataBase — Abstract base class for all magnet data containers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataType(IntEnum):
    PUPITRE = 0
    TDMS = 1
    ENSIGHT = 2
    HYBRID = 3


class MagnetDataBase(ABC):
    """Abstract base class for magnet data containers.

    Concrete subclasses implement data-format-specific logic; callers use the
    uniform interface defined here.

    Attributes
    ----------
    FileName : str
        Path or logical name of the source file.
    Groups : dict
        Format-specific group metadata (empty dict for pandas-backed data).
    Keys : list[str]
        Available channel/column names.
    units : dict
        Symbol/unit pairs populated by :meth:`Units`.
    """

    def __init__(
        self,
        filename: str,
        Groups: dict,
        Keys: list[str],
        Data: pd.DataFrame | dict | None = None,
    ) -> None:
        self.FileName = filename
        self.Groups = Groups
        self.Keys = Keys
        self.Data: pd.DataFrame | dict = Data if Data is not None else pd.DataFrame()
        self.units: dict = {}

    # ------------------------------------------------------------------
    # Abstract interface — every subclass must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def Type(self) -> DataType:
        """Data-type discriminator."""

    @abstractmethod
    def getData(self, key: list[str] | str | None = None) -> pd.DataFrame:
        """Return data for the given key(s)."""

    @abstractmethod
    def getKeys(self) -> list[str]:
        """Return list of available channel/column names."""

    @abstractmethod
    def Units(self, debug: bool = False) -> None:
        """Populate ``self.units`` with symbol/unit pairs."""

    @abstractmethod
    def getUnitKey(self, key: str) -> tuple:
        """Return ``(symbol, unit)`` for *key*."""

    @abstractmethod
    def extractData(self, keys: list[str]) -> pd.DataFrame:
        """Return a DataFrame with the requested columns/channels."""

    # ------------------------------------------------------------------
    # Concrete default implementations
    # ------------------------------------------------------------------

    def getType(self) -> DataType:
        """Return the data-type discriminator."""
        return self.Type

    def info(self) -> None:
        """Print a one-line summary."""
        logger.info(f"{self.__class__.__name__}: {self.FileName}")

    # Cleanup / reshape — no-op by default; pandas subclass overrides

    def cleanupData_legacy(self) -> int:  # noqa: N802
        return 0

    def cleanupData(  # noqa: N802
        self,
        keys_to_remove: list[str] | None = None,
        keys_to_rename: dict[str, str] | None = None,
        keys_to_add: dict[str, str] | None = None,
        debug: bool = False,
    ) -> int:
        return 0

    def removeData(self, keys: list) -> int:  # noqa: N802
        return 0

    @abstractmethod
    def renameData(self, columns: dict) -> None:  # noqa: N802
        ...

    # Compute / add — raise by default; subclasses override

    def addData(  # noqa: N802
        self, key: str, formula: str, unit: str | None = None, debug: bool = False
    ) -> int:
        raise NotImplementedError(f"{self.__class__.__name__}.addData not implemented")

    def computeData(  # noqa: N802
        self,
        method: Any,
        key: str,
        kparams: list,
        unit: tuple | None = None,
        debug: bool = False,
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.computeData not implemented")

    def saveData(self, keys: list[str], filename: str) -> int:  # noqa: N802
        raise NotImplementedError(f"{self.__class__.__name__}.saveData not implemented")

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
        raise NotImplementedError(f"{self.__class__.__name__}.plotData not implemented")

    def stats(self, key: str | None = None) -> pd.DataFrame | None:
        raise NotImplementedError(f"{self.__class__.__name__}.stats not implemented")

    # Time utilities — return empty / zero by default; subclasses override

    def getStartDate(self, group: str | None = None) -> tuple:  # noqa: N802
        return ()

    def getDuration(self, group: str | None = None) -> float:  # noqa: N802
        return 0.0

    def addTime(self) -> int:  # noqa: N802
        return 0

    def shiftTime(self, dt: float) -> int:  # noqa: N802
        return 0

    # Phase 2B hook — concrete implementations added in subclasses

    def get_time_range(self) -> tuple:
        """Return ``(start_datetime, end_datetime)`` for the dataset.

        Subclasses implementing time-aware formats should override this.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.get_time_range not implemented")

    # Convenience

    def extractDataThreshold(self, key: str, threshold: float) -> pd.DataFrame:  # noqa: N802
        raise NotImplementedError(
            f"{self.__class__.__name__}.extractDataThreshold not implemented"
        )

    def extractTimeData(  # noqa: N802
        self, timerange: str, group: str | None = None
    ) -> pd.DataFrame:
        raise NotImplementedError(f"{self.__class__.__name__}.extractTimeData not implemented")
