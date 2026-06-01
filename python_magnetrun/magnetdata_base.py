"""MagnetDataBase — Abstract base class for all magnet data containers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import IntEnum
from typing import Any, NamedTuple

import pandas as pd

from .utils.downsampling import DownsampleConfig

logger = logging.getLogger(__name__)


class FieldMeta(NamedTuple):
    """Physical metadata for one field/channel.

    Attributes
    ----------
    symbol:
        Short physical symbol used in axis labels (e.g. ``"B"``, ``"I"``).
    unit:
        Pint ``Unit`` object, or ``None`` for dimensionless / timestamp fields.
    label:
        Human-readable plot label (e.g. ``"Magnetic Field"``).  Empty string
        when not set in the JSON definition.
    description:
        Longer free-text description from the JSON file.
    """

    symbol: str
    unit: Any  # pint.Unit | None
    label: str
    description: str


class DataType(IntEnum):
    """Discriminator enum for the underlying data format.

    :cvar PUPITRE: pupitre ``.txt`` / ``.csv`` files backed by a pandas DataFrame.
    :cvar TDMS: pigbrother ``.tdms`` files backed by a dict of DataFrames.
    :cvar ENSIGHT: Ensight CSV files (pandas-backed, same as PUPITRE).
    :cvar HYBRID: hybrid data combining multiple sources.
    """

    PUPITRE = 0
    TDMS = 1
    ENSIGHT = 2
    HYBRID = 3


_ureg = None


def _make_ureg():  # type: ignore[return]
    """Return the shared pint UnitRegistry with project-specific units pre-registered.

    Returns a module-level singleton so that all Unit objects produced by this
    project share the same registry — a requirement for pint equality checks and
    for ``pd.concat`` to merge DataFrame attrs without raising
    "Cannot operate with Unit and Unit of different registries".
    """
    global _ureg
    if _ureg is not None:
        return _ureg

    from pint import UnitRegistry
    from pint.errors import UndefinedUnitError

    ureg = UnitRegistry()
    for defn, unit in [
        ("percent = 1 / 100 = %", "percent"),
        ("ppm = 1e-6 = ppm", "ppm"),
        ("var = 1", "var"),
    ]:
        try:
            ureg.parse_units(unit)
        except UndefinedUnitError:
            ureg.define(defn)
    _ureg = ureg
    return _ureg


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
    start_timestamp : datetime or None
        UTC timestamp of the first record in the dataset (naive, no tzinfo).
    end_timestamp : datetime or None
        UTC timestamp of the last record in the dataset (naive, no tzinfo).

    Timestamp convention
    --------------------
    All timestamp columns and ``start_timestamp`` / ``end_timestamp`` attributes
    store **naive UTC** :class:`~pandas.Timestamp` values (no tzinfo).

    ============================================================  ========================  =========
    Attribute / column                                            Type                      Value
    ============================================================  ========================  =========
    ``start_timestamp`` / ``end_timestamp``                       naive ``datetime``        UTC
    ``t`` column                                                  float                     elapsed seconds from first sample
    ``timestamp`` column                                          naive ``pd.Timestamp``    UTC
    ============================================================  ========================  =========

    Conversion to local time for display or user-facing filtering is performed
    **only** at display/filter boundaries (:meth:`plotData`,
    :meth:`extractTimeData`).  The ``time_zone`` parameter on those methods
    (default ``"Europe/Paris"``) controls the local timezone used.

    :meth:`addTime` must be called before :meth:`extractTimeData` or any
    ``timestamp``-based plot.  It is *eager*: it computes both ``t`` and
    ``timestamp`` for all groups at once.
    """

    def __init__(
        self,
        filename: str,
        Groups: dict,
        Keys: list[str],
        defs_file: str | None = None,
        start_timestamp: datetime | None = None,
        end_timestamp: datetime | None = None,
    ) -> None:
        """Initialise common attributes shared by all subclasses.

        :param filename: path or logical name of the source file.
        :param Groups: format-specific group metadata dict (empty for pandas
            data; keyed by group name for TDMS data).
        :param Keys: list of available channel/column names.
        :param defs_file: path to a JSON field-definition file used by
            :meth:`Units` to populate units; ``None`` disables JSON loading.
        :param start_timestamp: naive UTC datetime of the first sample; ``None``
            when not yet determined.
        :param end_timestamp: naive UTC datetime of the last sample; ``None``
            when not yet determined.
        """
        self.FileName = filename
        self.Groups = Groups
        self.Keys = Keys
        self.units: dict = {}
        self.field_meta: dict[str, FieldMeta] = {}
        self.defs_file: str | None = defs_file
        self.start_timestamp: datetime | None = start_timestamp
        self.end_timestamp: datetime | None = end_timestamp

    # ------------------------------------------------------------------
    # Abstract interface — every subclass must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def Data(self) -> pd.DataFrame | dict:
        """The loaded dataset.  Accessing this property triggers lazy loading."""

    @Data.setter
    @abstractmethod
    def Data(self, value: pd.DataFrame | dict) -> None:
        """Set the dataset backing store."""

    @property
    @abstractmethod
    def Type(self) -> DataType:
        """Data-type discriminator."""

    # ------------------------------------------------------------------
    # Resource lifecycle — concrete default; TDMS subclass overrides
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release any open file handles.  No-op by default."""
        return None

    def __enter__(self) -> MagnetDataBase:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @abstractmethod
    def getData(
        self,
        key: list[str] | str | None = None,
        downsample: DownsampleConfig | None = None,
    ) -> pd.DataFrame:
        """Return data for the given key(s), optionally downsampled."""

    @abstractmethod
    def getKeys(self) -> list[str]:
        """Return list of available channel/column names."""

    @abstractmethod
    def Units(self, debug: bool = False, json_file: str | None = None) -> None:
        """Populate ``self.units`` with symbol/unit pairs."""

    @abstractmethod
    def getUnitKey(self, key: str) -> tuple:
        """Return ``(symbol, unit)`` for *key*."""

    def getFieldMeta(self, key: str) -> FieldMeta | None:
        """Return :class:`FieldMeta` for *key*, or ``None`` if not available.

        Populated by :meth:`load_units_from_json`.  Returns ``None`` rather
        than raising so callers can fall back gracefully.
        """
        return self.field_meta.get(key)

    @abstractmethod
    def extractData(self, keys: list[str]) -> pd.DataFrame:
        """Return a DataFrame with the requested columns/channels."""

    # ------------------------------------------------------------------
    # Concrete default implementations
    # ------------------------------------------------------------------

    def load_units_from_json(self, json_file: str, debug: bool = False) -> None:
        """Populate ``self.units`` from a JSON field-definition file.

        Keys in the JSON that are not in ``self.Keys`` are silently ignored,
        so a single file can cover a superset of any particular dataset.

        JSON format — flat object, key = field name (or ``"Group/Channel"``
        for TDMS), value = ``{"symbol": ..., "unit": ..., "description": ...}``::

            {
                "Field":  {"symbol": "B", "unit": "tesla"},
                "Icoil1": {"symbol": "I", "unit": "ampere"},
                "Courants_Alimentations/Courant_A1": {"symbol": "I", "unit": "ampere"}
            }

        ``"unit": null`` stores ``None`` (used for timestamp/dimensionless).
        The ``"description"`` key is optional and only used for documentation.
        """
        from .field_defs import load_defs

        ureg = _make_ureg()

        field_defs: dict = load_defs(json_file)

        for key, defn in field_defs.items():
            if key.startswith("_") or key not in self.Keys:  # skip comment keys
                continue
            # if key not in self.Keys:
            #    logger.debug(f"load_units_from_json: {key!r} not in Keys, skipping")
            #    continue
            symbol: str = defn["symbol"]
            unit_str: str | None = defn.get("unit")
            if unit_str is None:
                pint_unit = None
            else:
                try:
                    parsed = ureg.parse_expression(unit_str)
                    # parse_expression may return a Quantity (e.g. 1 T) or a Unit;
                    # always store a Unit so formatting with ~P gives "T" not "1 T"
                    pint_unit = parsed.units if hasattr(parsed, "units") else parsed
                except (ValueError, AttributeError) as exc:
                    raise ValueError(
                        f"load_units_from_json: cannot parse unit {unit_str!r} for field {key!r}"
                    ) from exc
            label: str = defn.get("label", "")
            description: str = defn.get("description", "")
            self.units[key] = (symbol, pint_unit)
            self.field_meta[key] = FieldMeta(
                symbol=symbol, unit=pint_unit, label=label, description=description
            )
            if debug:
                logger.debug(
                    f"load_units_from_json: {key} → symbol={symbol}, unit={pint_unit}, label={label!r}"
                )

    def getType(self) -> DataType:
        """Return the data-type discriminator."""
        return self.Type

    def info(self) -> None:
        """Print a one-line summary."""
        logger.info(f"{self.__class__.__name__}: {self.FileName}")

    # Cleanup / reshape — no-op by default; pandas subclass overrides

    def cleanupData_legacy(self) -> int:  # noqa: N802
        """Legacy cleanup hook; no-op at this level.

        :returns: always ``0``.
        """
        return 0

    def cleanupData(  # noqa: N802
        self,
        keys_to_remove: list[str] | None = None,
        keys_to_rename: dict[str, str] | None = None,
        keys_to_add: dict[str, dict[str, Any]] | None = None,
        debug: bool = False,
    ) -> int:
        """Apply ETL transformations to the dataset (no-op at base level).

        Concrete subclasses override this to remove, rename, or add columns.

        :param keys_to_remove: column/channel names to drop.
        :param keys_to_rename: ``{old_name: new_name}`` mapping for renames.
        :param keys_to_add: ``{key: field_def}`` pairs for derived columns,
            where each *field_def* contains ``formula``, ``symbol``, ``unit``,
            ``label``, and ``description`` keys.
        :param debug: emit extra debug log messages when ``True``.
        :returns: ``0`` on success.
        """
        return 0

    def removeData(self, keys: list) -> int:  # noqa: N802
        """Remove columns from the dataset (no-op at base level).

        :param keys: list of column/channel names to drop.
        :returns: ``0`` on success.
        """
        return 0

    @abstractmethod
    def renameData(self, columns: dict) -> None:  # noqa: N802
        """Rename columns/channels in the dataset.

        :param columns: ``{old_name: new_name}`` mapping.
        """
        ...

    # Compute / add — raise by default; subclasses override

    def addData(  # noqa: N802
        self,
        key: str,
        formula: str,
        symbol: str,
        unit: Any,  # pint.Unit | str | None
        label: str,
        description: str,
        debug: bool = False,
    ) -> int:
        """Evaluate *formula* and add the result as a new column/channel *key*.

        :param key: name for the new column (or ``"Group/Channel"`` for TDMS).
        :param formula: pandas-compatible expression string (e.g.
            ``"Power = Icoil1 * Ucoil1"``).
        :param symbol: short physical symbol (e.g. ``"P"``).
        :param unit: pint ``Unit``, unit string, or ``None``.
        :param label: human-readable axis label.
        :param description: longer free-text description.
        :param debug: emit extra debug log messages when ``True``.
        :returns: ``0`` on success, non-zero on validation failure.
        :raises NotImplementedError: if not overridden by a subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.addData not implemented")

    def computeData(  # noqa: N802
        self,
        method: Any,
        key: str,
        kparams: list,
        symbol: str,
        unit: Any,  # pint.Unit | str | None
        label: str,
        description: str,
        debug: bool = False,
    ) -> int:
        """Apply a callable *method* row-wise and store the result as *key*.

        :param method: callable invoked as ``method(*row_values)`` for each row.
        :param key: name for the new column.
        :param kparams: list of existing column names passed as positional
            arguments to *method*.
        :param symbol: short physical symbol.
        :param unit: pint ``Unit``, unit string, or ``None``.
        :param label: human-readable axis label.
        :param description: longer free-text description.
        :param debug: emit extra debug log messages when ``True``.
        :returns: ``0`` on success, non-zero on failure.
        :raises NotImplementedError: if not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.computeData not implemented"
        )

    def saveData(self, keys: list[str], filename: str) -> int:  # noqa: N802
        """Save selected columns to *filename* as a tab-separated file.

        :param keys: column/channel names to export.
        :param filename: destination file path.
        :returns: ``0`` on success.
        :raises NotImplementedError: if not overridden by a subclass.
        """
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
        time_zone: str = "Europe/Paris",
        color: str | None = None,
        marker: str | None = None,
        linestyle: str | None = None,
        markevery: int | None = None,
    ) -> None:
        """Plot *y* versus *x* on a matplotlib *ax*.

        :param x: x-axis column/channel name (``"t"`` or ``"timestamp"``
            are also accepted).
        :param y: y-axis column/channel name.
        :param ax: matplotlib ``Axes`` object to draw on.
        :param alpha: line opacity (0–1, default ``1``).
        :param label: legend label; ``None`` uses the column name.
        :param normalize: divide *y* by its absolute maximum when ``True``.
        :param offset: unused offset parameter (kept for interface compatibility).
        :param time_zone: IANA timezone for local-time display of ``timestamp``
            columns (default ``"Europe/Paris"``).
        :param color: matplotlib colour string; ``None`` uses the default cycle.
        :param marker: matplotlib marker string; ``None`` uses no markers.
        :param linestyle: matplotlib linestyle string; ``None`` uses the default.
        :param markevery: plot a marker every *n* data points; ``None`` for
            every point.
        :raises NotImplementedError: if not overridden by a subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.plotData not implemented")

    def stats(self, key: str | None = None) -> pd.DataFrame | None:
        """Return or print descriptive statistics for the dataset.

        :param key: restrict statistics to this column/channel; ``None``
            computes stats for all columns.
        :returns: :class:`~pandas.DataFrame` of statistics, or ``None``.
        :raises NotImplementedError: if not overridden by a subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.stats not implemented")

    # Time utilities — return empty / zero by default; subclasses override

    def getStartDate(self, group: str | None = None) -> tuple:  # noqa: N802
        """Return ``(start_date, start_time, end_date, end_time)`` strings.

        :param group: group name (required for TDMS data; unused for pandas
            data).
        :returns: 4-tuple of strings in ``"%Y.%m.%d"`` / ``"%H:%M:%S"``
            format, or empty tuple when unavailable.
        """
        return ()

    def getDuration(self, group: str | None = None) -> float:  # noqa: N802
        """Return the duration of the dataset in seconds.

        :param group: group name (required for TDMS data; unused for pandas
            data).
        :returns: duration in seconds, or ``0.0`` when unavailable.
        """
        return 0.0

    def addTime(self, time_zone: str = "Europe/Paris") -> int:  # noqa: N802
        """Compute elapsed-time and absolute-timestamp columns (no-op at base level).

        :param time_zone: IANA timezone of the source date/time data.
        :returns: ``0`` on success.
        """
        return 0

    def shiftTime(self, dt: float) -> int:  # noqa: N802
        """Shift the ``t`` column by *dt* seconds (no-op at base level).

        :param dt: time offset in seconds to add to every ``t`` value.
        :returns: ``0`` on success.
        """
        return 0

    # Phase 2B hook — concrete implementations added in subclasses

    def get_time_range(self) -> tuple:
        """Return ``(start_timestamp, end_timestamp)`` for the dataset.

        Returns the stored :attr:`start_timestamp` / :attr:`end_timestamp`
        attributes when available.  Subclasses that derive timestamps from
        their data should override this and set those attributes accordingly.
        """
        if self.start_timestamp is None and self.end_timestamp is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.get_time_range not implemented"
            )
        return (self.start_timestamp, self.end_timestamp)

    # Convenience

    def extractDataThreshold(
        self, key: str, threshold: float
    ) -> pd.DataFrame:  # noqa: N802
        """Return rows where *key* ≥ *threshold*.

        :param key: column/channel name to filter on.
        :param threshold: minimum value (inclusive).
        :returns: filtered :class:`~pandas.DataFrame`.
        :raises NotImplementedError: if not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.extractDataThreshold not implemented"
        )

    def extractTimeData(  # noqa: N802
        self, timerange: str, group: str | None = None, time_zone: str = "Europe/Paris"
    ) -> pd.DataFrame:
        """Return rows whose ``timestamp`` falls within *timerange*.

        :param timerange: ``"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"``
            expressed in local time (*time_zone*).  Both boundaries are
            inclusive.
        :param group: TDMS group name (required for TDMS data; unused for
            pandas data).
        :param time_zone: IANA timezone of the datetime strings in *timerange*
            (default ``"Europe/Paris"``).
        :returns: filtered :class:`~pandas.DataFrame`.
        :raises NotImplementedError: if not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.extractTimeData not implemented"
        )
