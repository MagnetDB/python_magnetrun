import logging
from datetime import datetime
from typing import Any

import pandas as pd
import pint

from python_magnetrun.magnetdata_base import DataType, MagnetDataBase

logger = logging.getLogger(__name__)


class Signature:
    def __init__(
        self,
        name: str,
        symbol: str,
        unit: str,
        t0: datetime,
        timeshift: float,
        changes: list[int],
        regimes: list[str],
        times: list[float],
        values: list[float],
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
    def changes(self) -> list[int]:
        return self._changes

    @changes.setter
    def changes(self, value: list[int]):
        self._changes = value

    # Getter and setter for regimes
    @property
    def regimes(self) -> list[str]:
        return self._regimes

    @regimes.setter
    def regimes(self, value: list[str]):
        self._regimes = value

    # Getter and setter for times
    @property
    def times(self) -> list[float]:
        return self._times

    @times.setter
    def times(self, value: list[float]):
        self._times = value

    # Getter and setter for values
    @property
    def values(self) -> list[float]:
        return self._values

    @values.setter
    def values(self, value: list[float]):
        self._values = value

    def compact(self) -> None:
        """Merge consecutive regimes sharing the same tag into a single regime.

        ``changes``, ``regimes``, ``times`` and ``values`` are parallel lists
        where entry ``i`` marks the start of a regime segment tagged
        ``regimes[i]``. Because regime detection can split a run of the same
        tag on abrupt slope changes, the same tag may appear in several
        consecutive entries. This method collapses each such run down to its
        first entry, updating ``changes``, ``times`` and ``values`` in place
        to match.

        Returns
        -------
        None
            The instance is updated in place.
        """
        if not self._regimes:
            return

        new_changes = [self._changes[0]]
        new_regimes = [self._regimes[0]]
        new_times = [self._times[0]]
        new_values = [self._values[0]]

        for i in range(1, len(self._regimes)):
            if self._regimes[i] == new_regimes[-1]:
                continue
            new_changes.append(self._changes[i])
            new_regimes.append(self._regimes[i])
            new_times.append(self._times[i])
            new_values.append(self._values[i])

        self._changes = new_changes
        self._regimes = new_regimes
        self._times = new_times
        self._values = new_values

    @classmethod
    def from_df(
        cls,
        filename: str,
        t0: datetime,
        df: pd.DataFrame,
        key: str,
        symbol: str,
        unit: pint.Unit,
        tkey: str,
        threshold: float,
        timeshift: float = 0,
        show: bool = False,
        debug: bool = False,
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
            show=show,
            save=False,
            debug=debug,
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
        mdata: MagnetDataBase,
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
        if mdata.Type == DataType.TDMS:
            (group, channel) = key.split("/")
            t0 = mdata.Groups[group][channel]["wf_start_time"].astype(datetime)
            # print('tdms', type(t0), flush=True)
        else:
            t0 = mdata.start_timestamp
            # print('txt', type(t0), flush=True)

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

    def save(self, filename):
        """Save signature to file in JSON format"""
        import json
        import os

        sfilename = f"{filename}_signature.json"
        with open(filename, "w") as file:
            # Column names as strings
            column_names = ["time", self.name]

            # Method 1: Create DataFrame using a dictionary
            df = pd.DataFrame({column_names[0]: self.times, column_names[1]: self.values})
            csv_filename = f"{filename}-{self.name}.csv"
            df.to_csv(csv_filename, index=False)

            data = self.to_dict()
            data["csv_file"] = csv_filename
            del data["times"]
            del data["values"]
            json.dump(data, file, indent=4)
        logger.info(f"Signature saved to {os.path.abspath(sfilename)}")

    def plot(
        self,
        ax: Any | None = None,
        colors: Any | None = None,
        alpha: float = 0.2,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
    ) -> Any:
        """Plot the signature as a time series with regime-colored overlay spans.

        Draws the recorded change points (:attr:`times`, :attr:`values`) as a
        line and overlays colored :func:`~matplotlib.axes.Axes.axvspan`
        rectangles for each regime segment (``"U"``, ``"D"``, ``"P"``).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure and axes are created if ``None``.
        colors : PlotColors, optional
            Regime color configuration. Defaults to
            :data:`~python_magnetrun.plotting.style.DEFAULT_COLORS`
            (U=green, D=red, P=blue).
        alpha : float, optional
            Transparency of the regime overlay spans (default ``0.2``).
        show : bool, optional
            Call :func:`matplotlib.pyplot.show` after plotting (default ``False``).
        save : bool, optional
            Save the figure to *filename* (default ``False``).
        filename : str, optional
            Output path used when *save* is ``True``. Defaults to
            ``f"{self.name}_signature.png"``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the signature was plotted on.
        """
        import matplotlib.pyplot as plt

        from .analysis.plotting import plot_regimes
        from .plotting.style import DEFAULT_COLORS

        if ax is None:
            _fig, ax = plt.subplots()

        ax.plot(self.times, self.values, marker=".", label=self.name)
        plot_regimes(ax, self.regimes, self.times, colors=colors or DEFAULT_COLORS, alpha=alpha)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{self.symbol} [{self.unit}]")
        ax.set_title(f"Signature: {self.name}")
        ax.grid(True)
        ax.legend()

        if save:
            plt.savefig(filename or f"{self.name}_signature.png", dpi=300)
        if show:
            plt.show()

        return ax
