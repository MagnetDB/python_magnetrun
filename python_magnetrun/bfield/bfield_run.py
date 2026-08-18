"""BFieldRun — DataLoader-compatible wrapper for magnetic field profile data.

Wraps BProfileMagnetData and exposes it through the unified DataLoader protocol.
B-field profile data is inherently spatial (position vs. field), not temporal.
A measurement timestamp can be attached for cross-domain alignment purposes.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..magnetdata_pandas import BProfileMagnetData


class BFieldRun:
    """DataLoader-compatible wrapper for magnetic field profile data.

    Wraps :class:`~python_magnetrun.magnetdata_pandas.BProfileMagnetData`
    and exposes it through the unified ``DataLoader`` protocol.

    B-field profiles are spatial datasets ``(Position, Profile)``; they have
    no inherent time axis.  Pass *measurement_time* to associate the profile
    with a specific measurement instant so it can participate in time-based
    comparisons inside :class:`~python_magnetrun.comparison.session.ComparisonSession`.

    Parameters
    ----------
    data:
        Underlying :class:`~python_magnetrun.magnetdata_pandas.BProfileMagnetData`.
    housing:
        Housing identifier (e.g. ``"M8"``).
    assembly:
        Assembly identifier (e.g. ``"M9Ch1Lips"``).
    measurement_time:
        UTC datetime of the measurement.  When ``None``, :meth:`get_time_range`
        raises :exc:`NotImplementedError`.
    """

    def __init__(
        self,
        data: BProfileMagnetData,
        housing: str = "unknown",
        assembly: str = "",
        measurement_time: datetime | None = None,
    ) -> None:
        self._data = data
        self.Housing = housing
        self.Assembly = assembly
        self._measurement_time = measurement_time

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_bprofile(
        cls,
        filename: str,
        housing: str = "",
        assembly: str = "",
        measurement_time: datetime | None = None,
    ) -> BFieldRun:
        """Create from a B-profile CSV file (Index, Position, Profile columns)."""
        from ..magnetdata_pandas import BProfileMagnetData

        data = BProfileMagnetData.frombprofile(filename)
        return cls(data, housing=housing, assembly=assembly, measurement_time=measurement_time)

    # ------------------------------------------------------------------
    # DataLoader protocol
    # ------------------------------------------------------------------

    def getData(self, key: str | None = None) -> pd.DataFrame:
        return self._data.getData(key)

    def getKeys(self) -> list[str]:
        return self._data.getKeys()

    def getType(self) -> int:
        return int(self._data.Type)

    def getAssembly(self) -> str:
        return self.Assembly

    def getHousing(self) -> str:
        return self.Housing

    def getDomain(self) -> str:
        return "bfield"

    def get_time_range(self) -> tuple[datetime, datetime]:
        """Return ``(measurement_time, measurement_time)`` as a zero-duration range.

        Raises :exc:`NotImplementedError` when no *measurement_time* was
        provided — spatial-only profiles cannot be placed on a time axis.
        """
        if self._measurement_time is None:
            raise NotImplementedError(
                "BFieldRun has no time range — spatial data only. "
                "Pass measurement_time= to associate this profile with a timestamp."
            )
        return self._measurement_time, self._measurement_time
