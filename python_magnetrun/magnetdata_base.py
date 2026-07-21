"""MagnetDataBase — Abstract base class for all magnet data containers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import IntEnum
from typing import Any, NamedTuple

import numpy as np
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
    :cvar HTS: HTS files — ``;``-separated with units embedded in column headers.
    """

    PUPITRE = 0
    TDMS = 1
    ENSIGHT = 2
    HYBRID = 3
    HTS = 4


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

        Parameters
        ----------
        filename : str
            Path or logical name of the source file.
        Groups : dict
            Format-specific group metadata dict (empty for pandas data; keyed
            by group name for TDMS data).
        Keys : list[str]
            List of available channel/column names.
        defs_file : str, optional
            Path to a JSON field-definition file used by :meth:`Units` to
            populate units; ``None`` disables JSON loading.
        start_timestamp : datetime, optional
            Naive UTC datetime of the first sample; ``None`` when not yet
            determined.
        end_timestamp : datetime, optional
            Naive UTC datetime of the last sample; ``None`` when not yet
            determined.
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

    @abstractmethod
    def get_group_data(self, group: str) -> pd.DataFrame:
        """Return a DataFrame containing the time column and all channels in *group*.

        Parameters
        ----------
        group : str
            Group name as returned by :meth:`list_groups`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the time column (``"t"``) plus all columns that
            belong to *group* and are present in the loaded data.

        Raises
        ------
        KeyError
            If *group* is not in :attr:`Groups`.
        """

    # ------------------------------------------------------------------
    # Group management — concrete; TDMS subclass overrides mutating methods
    # ------------------------------------------------------------------

    def list_groups(self) -> list[str]:
        """Return the names of all defined groups.

        Returns
        -------
        list[str]
            Ordered list of group name strings.
        """
        return list(self.Groups.keys())

    def _validate_group_columns(self, columns: list[str]) -> None:
        """Validate that *columns* are in :attr:`Keys` and have required metadata.

        Parameters
        ----------
        columns : list[str]
            Column names to validate.

        Raises
        ------
        KeyError
            If any column is not in :attr:`Keys`.
        ValueError
            If any column has no entry in :attr:`field_meta` (i.e.
            :meth:`Units` has not been called) or has an empty symbol.
        """
        for col in columns:
            if col not in self.Keys:
                raise KeyError(
                    f"Column {col!r} is not in Keys. "
                    f"Available keys: {self.Keys}"
                )
            if col not in self.field_meta:
                raise ValueError(
                    f"Column {col!r} has no field metadata — "
                    "call Units() first or add an entry to the defs file."
                )
            meta = self.field_meta[col]
            if not meta.symbol:
                raise ValueError(
                    f"Column {col!r} has no symbol defined in field metadata."
                )
            if not meta.label:
                logger.warning(
                    f"Column {col!r} has no label defined (optional)."
                )

    def define_group(self, name: str, columns: list[str]) -> None:
        """Create or replace the group *name* with *columns*.

        Parameters
        ----------
        name : str
            Group name.
        columns : list[str]
            Column names to include; each must be in :attr:`Keys` and have
            ``symbol`` defined in :attr:`field_meta`.

        Raises
        ------
        KeyError
            If any column is not in :attr:`Keys`.
        ValueError
            If any column lacks required field metadata.
        """
        self._validate_group_columns(columns)
        self.Groups[name] = list(columns)

    def add_to_group(self, group: str, columns: str | list[str]) -> None:
        """Append *columns* to an existing group, creating it if absent.

        Parameters
        ----------
        group : str
            Group name.
        columns : str or list[str]
            Column name or list of column names to add.  Columns already
            present in the group are silently skipped (no duplicates).

        Raises
        ------
        KeyError
            If any column is not in :attr:`Keys`.
        ValueError
            If any column lacks required field metadata.
        """
        if isinstance(columns, str):
            columns = [columns]
        self._validate_group_columns(columns)
        existing: list[str] = self.Groups.get(group, [])
        self.Groups[group] = existing + [c for c in columns if c not in existing]

    def getFieldMeta(self, key: str) -> FieldMeta | None:
        """Return :class:`FieldMeta` for *key*, or ``None`` if not available.

        Populated by :meth:`load_units_from_json`.  Returns ``None`` rather
        than raising so callers can fall back gracefully.

        Parameters
        ----------
        key : str
            Channel/column name to look up.

        Returns
        -------
        FieldMeta or None
            Metadata for *key*, or ``None`` when not present.
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
        """Return the data-type discriminator.

        Returns
        -------
        DataType
            Data-type discriminator value.
        """
        return self.Type

    def info(self) -> None:
        """Print a one-line summary."""
        logger.info(f"{self.__class__.__name__}: {self.FileName}")

    # Cleanup / reshape — no-op by default; pandas subclass overrides

    def cleanupData_legacy(self) -> int:  # noqa: N802
        """Legacy cleanup hook; no-op at this level.

        Returns
        -------
        int
            Always ``0``.
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

        Parameters
        ----------
        keys_to_remove : list[str], optional
            Column/channel names to drop.
        keys_to_rename : dict[str, str], optional
            ``{old_name: new_name}`` mapping for renames.
        keys_to_add : dict[str, dict[str, Any]], optional
            ``{key: field_def}`` pairs for derived columns, where each
            *field_def* contains ``formula``, ``symbol``, ``unit``,
            ``label``, and ``description`` keys.
        debug : bool
            Emit extra debug log messages when ``True``.

        Returns
        -------
        int
            ``0`` on success.
        """
        return 0

    def removeData(self, keys: list) -> int:  # noqa: N802
        """Remove columns from the dataset (no-op at base level).

        Parameters
        ----------
        keys : list
            List of column/channel names to drop.

        Returns
        -------
        int
            ``0`` on success.
        """
        return 0

    @abstractmethod
    def renameData(self, columns: dict) -> None:  # noqa: N802
        """Rename columns/channels in the dataset.

        Parameters
        ----------
        columns : dict
            ``{old_name: new_name}`` mapping.
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

        Parameters
        ----------
        key : str
            Name for the new column (or ``"Group/Channel"`` for TDMS).
        formula : str
            Pandas-compatible expression string (e.g.
            ``"Power = Icoil1 * Ucoil1"``).
        symbol : str
            Short physical symbol (e.g. ``"P"``).
        unit : pint.Unit or str or None
            Pint ``Unit``, unit string, or ``None``.
        label : str
            Human-readable axis label.
        description : str
            Longer free-text description.
        debug : bool
            Emit extra debug log messages when ``True``.

        Returns
        -------
        int
            ``0`` on success, non-zero on validation failure.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
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

        Parameters
        ----------
        method : callable
            Callable invoked as ``method(*row_values)`` for each row.
        key : str
            Name for the new column.
        kparams : list
            Existing column names passed as positional arguments to *method*.
        symbol : str
            Short physical symbol.
        unit : pint.Unit or str or None
            Pint ``Unit``, unit string, or ``None``.
        label : str
            Human-readable axis label.
        description : str
            Longer free-text description.
        debug : bool
            Emit extra debug log messages when ``True``.

        Returns
        -------
        int
            ``0`` on success, non-zero on failure.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.computeData not implemented"
        )

    def add_field(
        self,
        key: str,
        values: list[float] | np.ndarray,
        symbol: str,
        unit: Any,  # pint.Unit | str | None
        label: str,
        description: str,
        group: str | None = None,
        debug: bool = False,
    ) -> int:
        """Store a pre-computed 1D array as a new field *key*.

        Parameters
        ----------
        key : str
            Name for the new column/channel (must not already exist).
        values : list[float] or numpy.ndarray
            1D sequence of values; length must match the number of existing
            rows.
        symbol : str
            Short physical symbol (e.g. ``"B"``).
        unit : pint.Unit or str or None
            Pint ``Unit``, unit string, or ``None``.
        label : str
            Human-readable axis label.
        description : str
            Longer free-text description.
        group : str, optional
            When given, the field is added to this group (created if it does
            not already exist) via :meth:`add_to_group`.
        debug : bool
            Emit extra debug log messages when ``True``.

        Returns
        -------
        int
            ``0`` on success, non-zero on validation failure.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.add_field not implemented")

    def saveData(self, keys: list[str], filename: str) -> int:  # noqa: N802
        """Save selected columns to *filename* as a tab-separated file.

        Parameters
        ----------
        keys : list[str]
            Column/channel names to export.
        filename : str
            Destination file path.

        Returns
        -------
        int
            ``0`` on success.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
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

        Parameters
        ----------
        x : str
            X-axis column/channel name; ``"t"`` and ``"timestamp"`` are also
            accepted.
        y : str
            Y-axis column/channel name.
        ax : matplotlib.axes.Axes
            Axes object to draw on.
        alpha : float
            Line opacity, 0–1 (default ``1``).
        label : str, optional
            Legend label; ``None`` uses the column name.
        normalize : bool
            Divide *y* by its absolute maximum when ``True``.
        offset : float
            Unused (kept for interface compatibility).
        time_zone : str
            IANA timezone for local-time display of ``timestamp`` columns
            (default ``"Europe/Paris"``).
        color : str, optional
            Matplotlib colour string; ``None`` uses the default cycle.
        marker : str, optional
            Matplotlib marker string; ``None`` uses no markers.
        linestyle : str, optional
            Matplotlib linestyle string; ``None`` uses the default.
        markevery : int, optional
            Plot a marker every *n* data points; ``None`` for every point.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.plotData not implemented")

    def stats(self, key: str | None = None) -> pd.DataFrame | None:
        """Return or print descriptive statistics for the dataset.

        Parameters
        ----------
        key : str, optional
            Restrict statistics to this column/channel; ``None`` computes
            stats for all columns.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame of statistics, or ``None``.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.stats not implemented")

    # Time utilities — return empty / zero by default; subclasses override

    def getStartDate(self, group: str | None = None) -> tuple:  # noqa: N802
        """Return ``(start_date, start_time, end_date, end_time)`` strings.

        Parameters
        ----------
        group : str, optional
            Group name (required for TDMS data; unused for pandas data).

        Returns
        -------
        tuple
            4-tuple of strings in ``"%Y.%m.%d"`` / ``"%H:%M:%S"`` format, or
            empty tuple when unavailable.
        """
        return ()

    def getDuration(self, group: str | None = None) -> float:  # noqa: N802
        """Return the duration of the dataset in seconds.

        Parameters
        ----------
        group : str, optional
            Group name (required for TDMS data; unused for pandas data).

        Returns
        -------
        float
            Duration [s], or ``0.0`` when unavailable.
        """
        return 0.0

    def addTime(self, time_zone: str = "Europe/Paris") -> int:  # noqa: N802
        """Compute elapsed-time and absolute-timestamp columns (no-op at base level).

        Parameters
        ----------
        time_zone : str
            IANA timezone of the source date/time data.

        Returns
        -------
        int
            ``0`` on success.
        """
        return 0

    def shiftTime(self, dt: float) -> int:  # noqa: N802
        """Shift the ``t`` column by *dt* seconds (no-op at base level).

        Parameters
        ----------
        dt : float
            Time offset [s] to add to every ``t`` value.

        Returns
        -------
        int
            ``0`` on success.
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

        Parameters
        ----------
        key : str
            Column/channel name to filter on.
        threshold : float
            Minimum value (inclusive).

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.extractDataThreshold not implemented"
        )

    def extractTimeData(  # noqa: N802
        self, timerange: str, group: str | None = None, time_zone: str = "Europe/Paris"
    ) -> pd.DataFrame:
        """Return rows whose ``timestamp`` falls within *timerange*.

        Parameters
        ----------
        timerange : str
            ``"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"`` expressed in local
            time (*time_zone*).  Both boundaries are inclusive.
        group : str, optional
            TDMS group name (required for TDMS data; unused for pandas data).
        time_zone : str
            IANA timezone of the datetime strings in *timerange* (default
            ``"Europe/Paris"``).

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.extractTimeData not implemented"
        )
