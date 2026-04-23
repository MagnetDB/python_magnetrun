"""SimulationRun — DataLoader-compatible wrapper for simulation data sources.

Wraps EnsightMagnetData, FeelppMagnetData, or MagnetToolsData and exposes
them through the unified DataLoader protocol.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..magnetdata_pandas import PandasMagnetData


class SimulationRun:
    """DataLoader-compatible wrapper for simulation data sources.

    Wraps :class:`~python_magnetrun.magnetdata_pandas.EnsightMagnetData`,
    :class:`~python_magnetrun.magnetdata_pandas.FeelppMagnetData`, or a
    magnettools-backed :class:`~python_magnetrun.magnetdata_pandas.PandasMagnetData`
    and exposes them through the unified ``DataLoader`` protocol.

    Parameters
    ----------
    data:
        Underlying pandas-backed data object.
    housing:
        Housing identifier (e.g. ``"M8"``).
    site:
        Site identifier (e.g. ``"grenoble"``).
    time_column:
        Column name to use as the time axis for :meth:`get_time_range`.
        Simulation data may be steady-state (no time column) or transient.
    """

    def __init__(
        self,
        data: PandasMagnetData,
        housing: str = "unknown",
        site: str = "",
        time_column: str | None = None,
    ) -> None:
        self._data = data
        self.Housing = housing
        self.Site = site
        self._time_column = time_column

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ensight(
        cls,
        filename: str,
        housing: str = "",
        site: str = "",
        time_column: str | None = None,
    ) -> SimulationRun:
        """Create from an Ensight CSV file."""
        from ..magnetdata_pandas import EnsightMagnetData

        data = EnsightMagnetData.fromensight(filename)
        return cls(data, housing=housing, site=site, time_column=time_column)

    @classmethod
    def from_feelpp(
        cls,
        filename: str,
        housing: str = "",
        site: str = "",
        skiprows: int = 0,
        time_column: str | None = None,
    ) -> SimulationRun:
        """Create from a Feel++ simulation CSV file."""
        from ..magnetdata_pandas import FeelppMagnetData

        data = FeelppMagnetData.fromfeelpp(filename, skiprows=skiprows)
        return cls(data, housing=housing, site=site, time_column=time_column)

    @classmethod
    def from_magnettools(
        cls,
        filename: str,
        housing: str = "",
        site: str = "",
        time_column: str | None = None,
    ) -> SimulationRun:
        """Create from a magnettools output file (stub — format TBD)."""
        from .magnettools_reader import load_magnettools

        data = load_magnettools(filename)
        return cls(data, housing=housing, site=site, time_column=time_column)

    # ------------------------------------------------------------------
    # DataLoader protocol
    # ------------------------------------------------------------------

    def getData(self, key: str | None = None) -> pd.DataFrame:
        return self._data.getData(key)

    def getKeys(self) -> list[str]:
        return self._data.getKeys()

    def getType(self) -> int:
        return int(self._data.Type)

    def getSite(self) -> str:
        return self.Site

    def getHousing(self) -> str:
        return self.Housing

    def getDomain(self) -> str:
        return "simulation"

    def get_time_range(self) -> tuple[datetime, datetime]:
        """Return ``(start, end)`` derived from *time_column*.

        Raises :exc:`NotImplementedError` if no time column was specified
        (steady-state / spatial simulation datasets have no time axis).
        """
        if self._time_column is None:
            raise NotImplementedError(
                "No time_column specified for this SimulationRun — "
                "steady-state datasets have no time axis."
            )
        df = self._data.getData()
        if self._time_column not in df.columns:
            raise KeyError(
                f"time_column {self._time_column!r} not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        t = pd.to_datetime(df[self._time_column])
        return t.min().to_pydatetime(), t.max().to_pydatetime()

    def is_transient(self) -> bool:
        """Return ``True`` when a time column is available."""
        return self._time_column is not None
