from datetime import datetime
from typing import List

import pandas as pd

from python_magnetrun.magnetdata import MagnetData


class Signature:
    def __init__(
        self,
        name: str,
        symbol: str,
        unit: str,
        t0: datetime,
        timeshift: float,
        changes: List[int],
        regimes: List[str],
        times: List[float],
        values: List[float],
    ):
        self._name = name
        self._symbol = symbol
        self._unit = unit
        self._t0 = t0
        self._timeshift = timeshift
        self._changes = changes
        self._regimes = regimes
        self._times = times
        self._values = values

    # Getter and setter for name
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    # Getter and setter for symbol
    @property
    def symbol(self) -> str:
        return self._symbol

    @symbol.setter
    def symbol(self, value: str):
        self._symbol = value

    # Getter and setter for unit
    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, value: str):
        self._unit = value

    # Getter and setter for t0
    @property
    def t0(self) -> datetime:
        return self._t0

    @t0.setter
    def t0(self, value: datetime):
        self._t0 = value

    # Getter and setter for timeshift
    @property
    def timeshift(self) -> float:
        return self._timeshift

    @timeshift.setter
    def timeshift(self, value: float):
        self._timeshift = value

    # Getter and setter for changes
    @property
    def changes(self) -> List[int]:
        return self._changes

    @changes.setter
    def changes(self, value: List[int]):
        self._changes = value

    # Getter and setter for regimes
    @property
    def regimes(self) -> List[str]:
        return self._regimes

    @regimes.setter
    def regimes(self, value: List[str]):
        self._regimes = value

    # Getter and setter for times
    @property
    def times(self) -> List[float]:
        return self._times

    @times.setter
    def times(self, value: List[float]):
        self._times = value

    # Getter and setter for values
    @property
    def values(self) -> List[float]:
        return self._values

    @values.setter
    def values(self, value: List[float]):
        self._values = value

    @classmethod
    def from_df(
        cls,
        filename: str,
        t0: datetime,
        df: pd.DataFrame,
        key: str,
        symbol: str,
        unit: str,
        tkey: str,
        threshold: float,
        timeshift: float = 0,
    ) -> "Signature":
        """
        Create a Signature instance from a magnetdata.

        Parameters:
        - df: pandas DataFrame containing 'regime', 'time', and 'value' columns.
        - key: key Field
        - symbol: Symbol of the signature.
        - unit: Unit of the signature.
        - tkey:
        - threshold: threshold for regime detection.
        - timeshift: lag in second

        Returns:
        - A Signature instance.
        """
        from .processing.trends import trends_df

        (changes, regimes, times, values, components) = trends_df(
            df,
            tkey,
            key,
            window=1,
            threshold=threshold,
            filename=filename,
            show=False,
            save=False,
            debug=False,
        )

        return cls(
            name=key,
            symbol=symbol,
            unit=f"{unit:~P}",
            t0=t0,
            timeshift=0,
            changes=changes,
            regimes=regimes,
            times=times,
            values=values,
        )

    @classmethod
    def from_mdata(
        cls,
        mdata: MagnetData,
        key: str,
        tkey: str,
        threshold: float,
        timeshift: float = 0,
    ) -> "Signature":
        """
        Create a Signature instance from a magnetdata.

        Parameters:
        - mdata: MagnetData
        - key: key Field
        - tkey:
        - threshold: threshold for regime detection.
        - timeshift: lag in second

        Returns:
        - A Signature instance.
        """
        from datetime import datetime
        from .processing.trends import trends

        t0 = datetime.now()
        (symbol, unit) = mdata.getUnitKey(key)
        match mdata.Data:
            case pd.DataFrame():
                t0 = mdata.Data.iloc[0]["timestamp"]
                # print('txt', type(t0), flush=True)
            case dict():
                (group, channel) = key.split("/")
                t0 = mdata.Groups[group][channel]["wf_start_time"].astype(datetime)
                # print('tdms', type(t0), flush=True)

        (changes, regimes, times, values, components) = trends(
            mdata,
            tkey,
            key,
            window=1,
            threshold=threshold,
            show=False,
            save=False,
            debug=False,
        )

        return cls(
            name=key,
            symbol=symbol,
            unit=f"{unit:~P}",
            t0=t0,
            timeshift=0,
            changes=changes,
            regimes=regimes,
            times=times,
            values=values,
        )

    def __repr__(self):
        return (
            f"Signature(name={self.name}, symbol={self.symbol}, unit={self.unit}, "
            f"t0={self.t0}, timeshit={self.timeshift}, changes={self.changes}, regimes={self.regimes}, times={self.times}, values={self.values})"
        )

    def to_dict(self):
        return {
            "name": self._name,
            "symbol": self._symbol,
            "unit": self._unit,
            "t0": self._t0,
            "timeshift": self._timeshift,
            "changes": self._changes,
            "regimes": self._regimes,
            "times": self._times,
            "values": self._values,
        }

    def dump(self, filename: str):
        import json

        with open(f"{filename}.json", "w") as file:
            # print('signature.dump:', self.to_dict())
            json.dump(self.to_dict(), file, indent=4)
