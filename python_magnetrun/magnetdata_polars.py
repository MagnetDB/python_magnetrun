"""PolarsMagnetData — Polars-backed magnet data (pupitre .txt files).

See ``prompts/polars-magnetdata-pupitre.plan.md`` for the design rationale
(sibling class to ``PandasMagnetData``, serving pupitre only) and scope
(Phase A: load/ETL parity with the ``MagnetRun.fromtxt`` -> ``prepareData``
call chain; analysis/plotting methods are Phase B, added on demand).

Not yet supported in this Phase A cut (documented, not silent):

- ``getData(downsample=...)`` — downsampling is not ported; raises
  ``NotImplementedError`` when a config is passed.
- Unit metadata is not attached to the ``getData()`` result the way
  ``PandasMagnetData`` attaches ``df.attrs["units"]`` — Polars/narwhals
  frames have no equivalent, and this is a plotting-facing concern
  (Phase B), not part of the ETL chain this class targets first.
- ``addData``/``computeData`` formulas: only ``+ - * /``, unary ``+ -``,
  parentheses, column names, and numeric literals are supported — the
  full grammar every real formula in this codebase's housing-config JSON
  files actually uses (simple column sums). Function calls (``sqrt``,
  ``sin``, …) are not supported; add them if a real formula ever needs one.
"""

from __future__ import annotations

import ast
import logging
import operator
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from natsort import natsorted

from .magnetdata_base import DataType, MagnetDataBase
from .utils.timestamps import parse_filename_timestamp
from .utils.timezone import (
    local_to_utc_naive,
    series_local_to_utc_naive,
    series_utc_to_local_naive,
    timerange_to_utc,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polars-native duplicate helpers (ports of utils/duplicates.py::find_duplicates
# and magnetdata_pandas.py::_get_duplicate_columns for a Polars backend)
# ---------------------------------------------------------------------------


def _polars_find_duplicates(
    df: pl.DataFrame, name: str, key: str, strict: bool = False
) -> pl.DataFrame:
    """Drop duplicate values of *key*, keeping the first occurrence.

    Polars equivalent of :func:`~python_magnetrun.utils.duplicates.find_duplicates`.
    """
    counts = df[key].value_counts()
    dup_counts = counts.filter(pl.col("count") > 1)
    if dup_counts.height > 0:
        total_duplicates = int((dup_counts["count"] - 1).sum())
        logger.warning(
            f"Duplicates found in {key}: {name} — {total_duplicates} duplicate(s) removed"
        )
        if strict:
            raise RuntimeError(f"Strict mode: duplicates found in {key} for {name}")
    return df.unique(subset=[key], keep="first", maintain_order=True)


def _polars_duplicate_columns(df: pl.DataFrame) -> list[str]:
    """Return column names that are exact duplicates of an earlier column."""
    duplicates: set[str] = set()
    columns = df.columns
    for x in range(df.width):
        col_x = df.to_series(x)
        for y in range(x + 1, df.width):
            if col_x.equals(df.to_series(y)):
                duplicates.add(columns[y])
    return list(duplicates)


# ---------------------------------------------------------------------------
# Formula evaluator — translates a "target = expr" formula's RHS into a
# Polars expression. See module docstring for the supported grammar.
# ---------------------------------------------------------------------------

_AST_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_AST_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _ast_to_polars_expr(node: ast.AST) -> pl.Expr | int | float:
    """Recursively translate one AST node into a Polars expression or literal."""
    if isinstance(node, ast.Expression):
        return _ast_to_polars_expr(node.body)
    if isinstance(node, ast.BinOp):
        op = _AST_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_ast_to_polars_expr(node.left), _ast_to_polars_expr(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _AST_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_ast_to_polars_expr(node.operand))
    if isinstance(node, ast.Name):
        return pl.col(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _formula_to_polars_expr(rhs: str) -> pl.Expr:
    """Parse a formula's right-hand side into a Polars expression."""
    tree = ast.parse(rhs.strip(), mode="eval")
    return _ast_to_polars_expr(tree)


class PolarsMagnetData(MagnetDataBase):
    """Polars-backed magnet data (pupitre ``.txt`` files only).

    ``self.Data`` is always a :class:`polars.DataFrame`.
    ``self.Type`` is ``DataType.PUPITRE``.
    """

    _TYPE: DataType = DataType.PUPITRE

    def __init__(
        self,
        filename: str,
        Groups: dict,
        Keys: list[str],
        Data: pl.DataFrame | None = None,
        defs_file: str | None = None,
        time_zone: str = "Europe/Paris",
    ) -> None:
        """Initialise a :class:`PolarsMagnetData` instance.

        Parameters
        ----------
        filename : str
            Path to the ``.txt`` file.
        Groups : dict
            Group metadata dict (always ``{}`` until :meth:`Units` populates
            it from a defs file's ``"group"`` key).
        Keys : list[str]
            List of column names.
        Data : polars.DataFrame, optional
            Pre-loaded DataFrame (typically a one-row stub); ``None`` enables
            lazy loading via :meth:`_ensure_data_loaded`.
        defs_file : str, optional
            Path to a JSON field-definition file; passed to
            :meth:`~.MagnetDataBase.Units`.
        time_zone : str
            IANA local timezone of the source ``Date``/``Time`` columns
            (default ``"Europe/Paris"``); used to convert ``start_timestamp``
            to naive UTC.
        """
        self._data: pl.DataFrame = Data if isinstance(Data, pl.DataFrame) else pl.DataFrame()
        self._data_loaded: bool = self._data.height > 1
        super().__init__(filename, Groups, Keys, defs_file=defs_file)
        dt = parse_filename_timestamp(filename)  # in local time
        self.start_timestamp = pd.Timestamp(dt) if dt is not None else None
        self._validate_start_timestamp()
        if self.start_timestamp is not None:
            self.start_timestamp = local_to_utc_naive(self.start_timestamp, time_zone)

    # --- Data property (implements lazy loading) ---------------------

    @property
    def Data(self) -> pl.DataFrame:
        self._ensure_data_loaded()
        return self._data

    @Data.setter
    def Data(self, value: pl.DataFrame | dict) -> None:
        if isinstance(value, dict):
            raise ValueError(
                "Data setter: dict value not supported for PolarsMagnetData; expected a polars DataFrame"
            )
        self._data = value

    def _ensure_data_loaded(self) -> None:
        """Load the full file from disk on first data access.

        Subsequent calls are no-ops.
        """
        if self._data_loaded:
            return
        from .readers.csv_readers import PupitreReader
        from .utils.validation import FileFormatError

        df = PupitreReader().read_polars(self.FileName)
        if df.height == 0:
            raise FileFormatError(
                f"{self.FileName}: no data rows found (header-only file)"
            )
        self._data_loaded = True  # set before assigning self.Data to avoid recursion
        self.Data = df
        self.Keys = df.columns
        logger.debug(f"_ensure_data_loaded: loaded {self.FileName} ({df.height} rows)")

    # --- abstract property --------------------------------------------

    @property
    def Type(self) -> DataType:
        return self._TYPE

    # --- core data access ----------------------------------------------

    def getPolarsData(self, key: list[str] | str | None) -> pl.DataFrame:
        """Return the full DataFrame or a column selection.

        Parameters
        ----------
        key : str or list[str] or None
            Column name, list of column names, or ``None`` for the full
            DataFrame.

        Returns
        -------
        polars.DataFrame
            DataFrame for the requested key(s).

        Raises
        ------
        KeyError
            If any requested key is not in :attr:`Keys`.
        """
        self._ensure_data_loaded()
        if key is None:
            return self.Data
        selected_keys = key if isinstance(key, list) else [key]
        for item in selected_keys:
            if item not in self.Keys:
                raise KeyError(
                    f"MagnetData/Data({key}): {self.FileName}: cannot get data for key={item}: no such key"
                )
        return self.Data.select(selected_keys)

    def getData(
        self,
        key: list[str] | str | None = None,
        downsample: Any = None,
    ) -> Any:
        """Return data for the given key(s), wrapped as a narwhals frame.

        The narwhals boundary lets callers work with a backend-agnostic
        frame regardless of whether the underlying container is Polars
        (this class) or pandas (:class:`PandasMagnetData`) — see
        ``prompts/mrun-cache-implementation.plan.md`` Phase 2.

        Parameters
        ----------
        key : str or list[str] or None
            Column name, list of column names, or ``None`` for all columns.
        downsample : DownsampleConfig, optional
            Not yet supported for this backend; passing a value raises.

        Returns
        -------
        narwhals.DataFrame
            Requested data, wrapped with :func:`narwhals.from_native`. Call
            ``.to_native()`` to get the underlying :class:`polars.DataFrame`.

        Raises
        ------
        NotImplementedError
            If *downsample* is not ``None`` — see module docstring.
        """
        import narwhals as nw

        if downsample is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.getData: downsample is not yet supported"
            )
        return nw.from_native(self.getPolarsData(key), eager_only=True)

    def getKeys(self) -> list[str]:
        """Return the list of available column names."""
        return self.Keys

    # --- units -----------------------------------------------------------
    # Backend-agnostic: operates on self.Keys / a JSON defs file only, never
    # touches self.Data — copied verbatim from PandasMagnetData.

    def Units(
        self, debug: bool = False, json_file: str | None = None
    ) -> None:  # noqa: N802
        """Populate ``self.units`` from column names.

        Resolution order:
        1. *json_file* argument (explicit override)
        2. ``self.defs_file`` set at construction time
        3. Built-in pattern matching (fallback, kept for backward compatibility)
        """
        from .magnetdata_base import _make_ureg

        resolved = json_file or self.defs_file
        if resolved is not None:
            self.load_units_from_json(resolved, debug=debug)
            self._build_groups(resolved)

        ureg = _make_ureg()

        for key in self.Keys:
            if key in self.units:
                continue
            logger.warning(
                f"Units: no JSON definition for key '{key}', applying legacy pattern matching"
            )
            if key in ("Date", "Time"):
                pass
            elif key == "timestamp":
                self.units[key] = ("time", None)
            elif key == "t":
                self.units[key] = ("t", ureg.second)
            elif key == "Field":
                self.units[key] = ("B", ureg.tesla)
            elif key.startswith("I"):
                self.units[key] = ("I", ureg.ampere)
            elif key.startswith("U"):
                self.units[key] = ("U", ureg.volt)
            elif key.startswith("T") or key == "teb" or key == "tsb":
                self.units[key] = ("T", ureg.degC)
            elif key.startswith("Rpm"):
                self.units[key] = ("Rpm", ureg.rpm)
            elif key.startswith("DR"):
                self.units[key] = ("%", ureg.percent)
            elif key.startswith("Flo"):
                self.units[key] = ("Q", ureg.liter / ureg.second)
            elif key == "debitbrut":
                self.units[key] = ("Q", ureg.meter**3 / ureg.hour)
            elif key.startswith("HP") or key.startswith("BP"):
                self.units[key] = ("P", ureg.bar)
            elif key == "Pmagnet" or key == "Ptot" or key.startswith("Power"):
                self.units[key] = ("Power", ureg.megawatt)
            elif key == "Q":
                self.units[key] = ("Preac", ureg.megavar)
            else:
                logger.warning(f"Units: no unit defined for key '{key}' — skipping")

    def getUnitKey(self, key: str) -> tuple:
        """Return the ``(symbol, unit)`` pair for *key*."""
        if key not in self.Keys:
            from .magnetdata_base import _make_ureg

            ureg = _make_ureg()
            if key == "t":
                return ("t", ureg.second)
            elif key == "timestamp":
                return ("time", None)
            else:
                raise RuntimeError(
                    f"{key} not defined in data - available keys are {self.Keys}"
                )
        return self.units[key]

    def _build_groups(self, json_file: str) -> None:
        """Populate :attr:`Groups` from the ``"group"`` key in the defs file."""
        from .field_defs import load_defs

        groups: dict[str, list[str]] = {}
        for key, defn in load_defs(json_file).items():
            if key.startswith("_") or key not in self.Keys:
                continue
            grp = defn.get("group")
            if grp:
                groups.setdefault(grp, []).append(key)
        self.Groups = groups

    def get_group_data(self, group: str) -> pl.DataFrame:
        """Return a DataFrame with the time column and all channels in *group*."""
        if group not in self.Groups:
            raise KeyError(
                f"Group {group!r} not found. Available groups: {self.list_groups()}"
            )
        cols = ["t", "timestamp"] + self.Groups[group]
        return self.Data.select([c for c in cols if c in self.Data.columns])

    # --- timestamp validation --------------------------------------------

    def _validate_start_timestamp(self) -> None:
        """Cross-check ``start_timestamp`` against the first ``Date``/``Time`` data row."""
        if "Date" not in self.Keys or "Time" not in self.Keys:
            return
        df = self._data  # bypass property — stub already has row 0, no full load needed
        if not isinstance(df, pl.DataFrame) or df.height == 0:
            return
        try:
            date_str = str(df["Date"][0])
            time_str = str(df["Time"][0])
            data_ts = pd.Timestamp(
                datetime.strptime(f"{date_str} {time_str}", "%Y.%m.%d %H:%M:%S")
            )
        except (ValueError, TypeError):
            logger.warning(
                f"_validate_start_timestamp: cannot parse Date/Time from first row of {self.FileName!r}"
            )
            return

        if self.start_timestamp is None:
            self.start_timestamp = data_ts
        elif self.start_timestamp != data_ts:
            logger.info(
                f"_validate_start_timestamp: {self.FileName!r} — filename timestamp {self.start_timestamp} "
                f"differs from data timestamp {data_ts}; using data value -- aka {data_ts}"
            )
            self.start_timestamp = data_ts

    # --- cleanup / reshape -------------------------------------------------

    def cleanupData(  # noqa: N802
        self,
        keys_to_remove: list[str] | None = None,
        keys_to_rename: dict[str, str] | None = None,
        keys_to_add: dict[str, dict[str, Any]] | None = None,
        debug: bool = False,
    ) -> int:
        """Apply ETL transformations (add, rename, remove columns) and normalise the DataFrame.

        Same behaviour as :meth:`PandasMagnetData.cleanupData`, ported to
        Polars. See that method's docstring for the parameter contract.
        """
        self._ensure_data_loaded()
        logger.debug(f"Clean up Data: filename={self.FileName}, keys={self.Keys}")
        assert isinstance(self.Data, pl.DataFrame)

        if keys_to_add:
            logger.debug(f"cleanupData: adding keys {list(keys_to_add.keys())}")
            existing_keys = [key for key in keys_to_add if key in self.Keys]
            if existing_keys:
                logger.warning(
                    f"cleanupData: keys {existing_keys} already exist in DataFrame, skipping addition"
                )
            for key, field_def in keys_to_add.items():
                status = self.addData(
                    key,
                    field_def["formula"],
                    symbol=field_def["symbol"],
                    unit=field_def["unit"],
                    label=field_def["label"],
                    description=field_def["description"],
                    debug=debug,
                )
                if status != 0:
                    logger.warning(
                        f"cleanupData: failed to add {key!r} (status={status})"
                    )

        if keys_to_rename:
            logger.debug(f"cleanupData: renaming keys {keys_to_rename}")
            missing_keys = [key for key in keys_to_rename if key not in self.Keys]
            if missing_keys:
                logger.warning(
                    f"cleanupData: keys {missing_keys} not found in DataFrame, cannot rename"
                )
            target_exists = [
                new_key for new_key in keys_to_rename.values() if new_key in self.Keys
            ]
            if target_exists:
                logger.warning(
                    f"cleanupData: target keys {target_exists} already exist in DataFrame, will be overwritten"
                )
            self.renameData(keys_to_rename)

        if keys_to_remove:
            logger.debug(f"cleanupData: removing keys {keys_to_remove}")
            missing_keys = [key for key in keys_to_remove if key not in self.Keys]
            if missing_keys:
                logger.warning(
                    f"cleanupData: keys {missing_keys} not found in DataFrame, cannot remove"
                )
            self.removeData(keys_to_remove)

        self.Keys = self.Data.columns

        if "t" in self.Keys:
            self.Data = _polars_find_duplicates(self.Data, self.FileName, "t")
            self.Keys = self.Data.columns

        Fkeys = set(
            [_key for _key in self.Keys if re.match(r"Flow\w+", _key)]
            + [_key for _key in self.Keys if re.match(r"Rpm\w+", _key)]
            + [_key for _key in self.Keys if re.match(r"HP\w+", _key)]
            + [_key for _key in self.Keys if re.match(r"\w+_ref", _key)]
            + [_key for _key in self.Keys if re.match(r"Pmagnet", _key)]
            + [_key for _key in self.Keys if re.match(r"Ptot", _key)]
            + [_key for _key in self.Keys if re.match(r"Idcct\d", _key)]
            + [_key for _key in self.Keys if re.match(r"IH$|IB$", _key)]
            + [
                _key
                for _key in self.Keys
                if re.match(r"(Supra)?Field|TotalField", _key)
            ]
            + [_key for _key in self.Keys if re.match(r"TAlimout", _key)]
        )

        # Only numeric columns are candidates for the all-zero check —
        # comparing a non-numeric (string/datetime) column to 0 raises in
        # Polars, unlike pandas which silently evaluates to all-False.
        numeric_cols = [
            name
            for name, dtype in zip(self.Data.columns, self.Data.dtypes, strict=True)
            if dtype.is_numeric()
        ]
        zero_cols = [c for c in numeric_cols if bool((self.Data[c] == 0).all())]
        logger.debug(f"zero columns: {natsorted(zero_cols)}")

        empty_cols = [col for col in zero_cols if col not in Fkeys]
        logger.info(f"empty cols (to drop): {natsorted(empty_cols)}")
        if empty_cols:
            self.Data = self.Data.drop(empty_cols)
            self.Keys = self.Data.columns

        dropped_columns = _polars_duplicate_columns(self.Data)
        really_dropped_columns = natsorted(
            [
                col
                for col in dropped_columns
                if not col.startswith("Ucoil") and col not in Fkeys
            ]
        )
        logger.info(
            f"duplicate columns (others than Ucoil* and Fkeys): {natsorted(really_dropped_columns)}"
        )
        if really_dropped_columns:
            self.Data = self.Data.drop(really_dropped_columns)
            self.Keys = self.Data.columns

        return 0

    def removeData(self, keys: list) -> int:  # noqa: N802
        """Drop columns from the DataFrame."""
        assert isinstance(self.Data, pl.DataFrame)
        existing = [key for key in keys if key in self.Keys]
        for key in keys:
            if key not in self.Keys:
                logger.warning(
                    f"removeData: cannot remove '{key}', key not found - skipping"
                )
        if existing:
            self.Data = self.Data.drop(existing)
        self.Keys = self.Data.columns
        return 0

    def renameData(self, columns: dict) -> None:  # noqa: N802
        """Rename DataFrame columns."""
        assert isinstance(self.Data, pl.DataFrame)
        missing = [old for old in columns if old not in self.Keys]
        if missing:
            logger.warning(
                f"renameData: keys {missing} not found in DataFrame, skipping"
            )
            columns = {old: new for old, new in columns.items() if old not in missing}
        if not columns:
            return
        source_keys = set(columns.keys())
        conflicts = {
            old: new
            for old, new in columns.items()
            if new in self.Keys and new not in source_keys
        }
        if conflicts:
            raise ValueError(
                f"renameData: target name(s) already exist in DataFrame and would "
                f"be silently overwritten: {conflicts}"
            )
        self.Data = self.Data.rename(columns)
        self.Keys = self.Data.columns

    # --- compute / add -----------------------------------------------------

    def _validate_formula_keys(self, formula: str) -> tuple[int, str]:
        """Validate that all variables referenced in a formula exist in the DataFrame."""
        assert isinstance(self.Data, pl.DataFrame)
        status = 0

        rhs = formula.split("=", 1)[1] if "=" in formula else formula
        remaining_vars = set(re.findall(r"\b([a-zA-Z_]\w*)\b", rhs))
        undefined_vars = [v for v in remaining_vars if v not in self.Data.columns]
        if undefined_vars:
            logger.warning(
                f"_validate_formula_keys: {rhs}: undefined variables:\n"
                f"  - Variables not found: {undefined_vars}\n"
                f"  - Available columns: {self.Data.columns}"
            )
            status = 1
        return status, formula

    def _validate_kparams(self, kparams: list) -> tuple[int, list]:
        """Validate that all parameter keys exist in the DataFrame."""
        assert isinstance(self.Data, pl.DataFrame)
        status = 0
        missing_params = [p for p in kparams if p not in self.Data.columns]
        if missing_params:
            logger.warning(
                "_validate_kparams: missing parameters:\n"
                f"  - Missing: {missing_params}\n"
                f"  - Available columns: {self.Data.columns}"
            )
            status = 1
        return status, kparams

    def addData(  # noqa: N802
        self,
        key: str,
        formula: str,
        symbol: str,
        unit: Any,
        label: str,
        description: str,
        debug: bool = False,
    ) -> int:
        """Evaluate *formula* and add the result as a new column *key*.

        See module docstring for the supported formula grammar (arithmetic
        only — ``+ - * /``, no function calls).
        """
        from pint.errors import UndefinedUnitError

        from .magnetdata_base import FieldMeta, _make_ureg

        assert isinstance(self.Data, pl.DataFrame)
        if key in self.Keys:
            logger.warning(
                f"addData: key '{key}' already exists in DataFrame, skipping addition"
            )
            return 1

        status, formula = self._validate_formula_keys(formula)
        if status != 0:
            logger.warning(
                f"addData: {key}: formula validation returned status {status}; "
                f"skipping evaluation"
            )
            return status

        rhs = formula.split("=", 1)[1] if "=" in formula else formula
        try:
            expr = _formula_to_polars_expr(rhs)
        except (SyntaxError, ValueError) as exc:
            logger.warning(f"addData: {key}: cannot parse formula {formula!r}: {exc}")
            return 1

        self.Data = self.Data.with_columns(expr.alias(key))
        self.Keys = self.Data.columns

        if isinstance(unit, str) and unit:
            try:
                ureg = _make_ureg()
                parsed = ureg.parse_expression(unit)
                pint_unit = parsed.units if hasattr(parsed, "units") else parsed
            except (ValueError, UndefinedUnitError):
                pint_unit = None
        else:
            pint_unit = unit if unit else None

        self.units[key] = (symbol, pint_unit)
        self.field_meta[key] = FieldMeta(
            symbol=symbol, unit=pint_unit, label=label, description=description
        )
        return 0

    def computeData(  # noqa: N802
        self,
        method: Any,
        key: str,
        kparams: list,
        symbol: str,
        unit: Any,
        label: str,
        description: str,
        debug: bool = False,
    ) -> int:
        """Apply *method* row-wise over *kparams* columns and store the result as *key*."""
        from pint.errors import UndefinedUnitError

        from .magnetdata_base import FieldMeta, _make_ureg

        if key in self.Keys:
            logger.warning(f"Key {key} already exists in DataFrame")
            return 1

        status, kparams = self._validate_kparams(kparams)
        if status != 0:
            logger.warning(
                f"computeData: {key}: kparams validation returned status {status}; "
                f"skipping computation"
            )
            return status

        assert isinstance(self.Data, pl.DataFrame)
        data = [method(*values) for values in self.Data.select(kparams).iter_rows()]
        self.Data = self.Data.with_columns(pl.Series(key, data))
        self.Keys = self.Data.columns

        if isinstance(unit, str) and unit:
            try:
                ureg = _make_ureg()
                parsed = ureg.parse_expression(unit)
                pint_unit = parsed.units if hasattr(parsed, "units") else parsed
            except (ValueError, UndefinedUnitError):
                pint_unit = None
        else:
            pint_unit = unit if unit else None
        self.units[key] = (symbol, pint_unit)
        self.field_meta[key] = FieldMeta(
            symbol=symbol, unit=pint_unit, label=label, description=description
        )
        return 0

    # --- time utilities ------------------------------------------------

    def getStartDate(self, group: str | None = None) -> tuple:  # noqa: N802
        """Return start/end date and time strings from the ``Date``/``Time`` columns."""
        res: tuple = ()
        if "Date" in self.Keys and "Time" in self.Keys:
            start_date = self.Data["Date"][0]
            start_time = self.Data["Time"][0]
            end_date = self.Data["Date"][-1]
            end_time = self.Data["Time"][-1]
            res = (start_date, start_time, end_date, end_time)
        return res

    def getDuration(self, group: str | None = None) -> float:  # noqa: N802
        """Return the duration of the dataset in seconds."""
        if "t" in self.Keys:
            assert isinstance(self.Data, pl.DataFrame)
            return float(self.Data["t"][-1] - self.Data["t"][0])
        logger.warning("magnetdata.getDuration: no t key")
        logger.warning(f"available keys are: {self.Keys}")
        return 0.0

    def addTime(self, time_zone: str = "Europe/Paris") -> int:  # noqa: N802
        """Compute ``t`` (elapsed seconds) and ``timestamp`` (naive UTC) columns.

        Drops ``Date`` and ``Time`` after conversion. The DST-aware
        local-to-UTC conversion reuses the already-vetted pandas
        implementation (:func:`~.utils.timezone.series_local_to_utc_naive`,
        which handles DST-ambiguity edge cases) via a small, one-time round
        trip through a pandas Series — this runs once per file, not in a hot
        loop, so the conversion cost is negligible.

        Parameters
        ----------
        time_zone : str
            IANA timezone of the source ``Date``/``Time`` columns (default
            ``"Europe/Paris"``).

        Returns
        -------
        int
            ``0`` on success.
        """
        self._ensure_data_loaded()
        assert isinstance(self.Data, pl.DataFrame)
        if "Date" not in self.Keys or "Time" not in self.Keys:
            raise RuntimeError(
                f"MagnetData/AddTime {self.FileName}: cannot add t[s] columnn: no Date or Time columns"
            )

        try:
            df = self.Data.with_columns(
                (
                    pl.col("Date").str.strptime(pl.Date, "%Y.%m.%d").cast(pl.Datetime("us"))
                    + pl.col("Time").str.strptime(pl.Time, "%H:%M:%S").cast(pl.Duration("us"))
                ).alias("_timestamp")
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"MagnetData/AddTime {self.FileName}: failed to create timestamp column"
            ) from exc

        df = _polars_find_duplicates(df, self.FileName, "_timestamp")

        t0 = df["_timestamp"][0]
        df = df.with_columns(
            (pl.col("_timestamp") - t0).dt.total_seconds().cast(pl.Float64).alias("t")
        )

        utc_series = series_local_to_utc_naive(df["_timestamp"].to_pandas(), time_zone)
        df = df.with_columns(pl.from_pandas(utc_series).alias("timestamp"))

        df = df.drop(["Date", "Time", "_timestamp"])
        self.Data = df
        self.Keys = self.Data.columns
        return 0

    def shiftTime(self, dt: float) -> int:  # noqa: N802
        """Shift the ``t`` column by *dt* seconds."""
        if "t" in self.Keys:
            self.Data = self.Data.with_columns((pl.col("t") + dt).alias("t"))
        else:
            raise RuntimeError(
                f"MagnetData/shiftTime {self.FileName}: cannot shift t[s] columnn: no t column"
            )
        return 0

    def get_time_range(self) -> tuple:
        """Return ``(start_timestamp, end_timestamp)`` for the dataset."""
        if self.start_timestamp is not None:
            duration = self.getDuration()
            self.end_timestamp = self.start_timestamp + pd.Timedelta(seconds=duration)
            return (self.start_timestamp, self.end_timestamp)

        if "Date" not in self.Keys or "Time" not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.get_time_range: no Date/Time columns in {self.FileName}"
            )
        assert isinstance(self.Data, pl.DataFrame)
        tformat = "%Y.%m.%d %H:%M:%S"
        start_str = f"{self.Data['Date'][0]} {self.Data['Time'][0]}"
        end_str = f"{self.Data['Date'][-1]} {self.Data['Time'][-1]}"
        self.start_timestamp = pd.Timestamp(datetime.strptime(start_str, tformat))
        self.end_timestamp = pd.Timestamp(datetime.strptime(end_str, tformat))
        return (self.start_timestamp, self.end_timestamp)

    # --- extract ---------------------------------------------------------

    def extractData(self, keys: list[str]) -> pl.DataFrame:  # noqa: N802
        """Return a DataFrame containing only the requested columns."""
        for key in keys:
            if key not in self.Keys:
                raise RuntimeError(f"{self.__class__.__name__}.extractData: no {key} key")
        return self.Data.select(keys)

    def extractDataThreshold(  # noqa: N802
        self, key: str, threshold: float
    ) -> pl.DataFrame:
        """Return rows where *key* >= *threshold*.

        Parameters
        ----------
        key : str
            Column name to filter on.
        threshold : float
            Minimum value (inclusive).

        Returns
        -------
        polars.DataFrame
            Filtered DataFrame.

        Raises
        ------
        RuntimeError
            If *key* is not present in :attr:`Keys`.
        """
        if key not in self.Keys:
            raise RuntimeError(
                f"extractData: key={key} - no such keys in dataframe (valid keys are: {self.Keys}"
            )
        return self.Data.filter(pl.col(key) >= threshold)

    def extractTimeData(  # noqa: N802
        self, timerange: str, group: str | None = None, time_zone: str = "Europe/Paris"
    ) -> pl.DataFrame:
        """Return rows whose ``timestamp`` falls within *timerange*.

        Parameters
        ----------
        timerange : str
            ``"YYYY-MM-DD HH:MM:SS;YYYY-MM-DD HH:MM:SS"`` in local time (the
            *time_zone* timezone). Both boundaries are inclusive.
        group : str, optional
            Unused; accepted for interface compatibility.
        time_zone : str
            IANA timezone of the datetime strings in *timerange* (default
            ``"Europe/Paris"``).

        Returns
        -------
        polars.DataFrame
            Filtered DataFrame.

        Raises
        ------
        RuntimeError
            If :meth:`addTime` has not been called yet.
        """
        if "timestamp" not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.extractTimeData: call addTime() before extractTimeData()"
            )
        logger.debug(f"Select data from {timerange}")
        t_start, t_end = timerange_to_utc(timerange, time_zone)
        return self.Data.filter(pl.col("timestamp").is_between(t_start, t_end, closed="both"))

    # --- persist / display -------------------------------------------------

    def saveData(self, keys: list[str], filename: str) -> int:  # noqa: N802
        """Save selected columns to *filename* as a tab-separated file."""
        self.Data.select(keys).write_csv(filename, separator="\t", include_header=True)
        return 0

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

        Unlike :meth:`PandasMagnetData.plotData`, this does not go through
        pandas' ``DataFrame.plot()`` (Polars has no equivalent) — the line is
        drawn directly via ``ax.plot()`` on the underlying numpy arrays.

        Parameters
        ----------
        x : str
            X-axis column name; ``"t"`` and ``"timestamp"`` are also accepted.
        y : str
            Y-axis column name.
        ax : matplotlib.axes.Axes
            Axes object to draw on.
        alpha : float
            Line opacity, 0-1 (default ``1``).
        label : str, optional
            Legend label; ``None`` uses the column name.
        normalize : bool
            Divide *y* by its absolute maximum when ``True``.
        offset : float
            Unused (kept for interface compatibility).
        time_zone : str
            IANA timezone for local-time display of ``"timestamp"`` x-axis
            (default ``"Europe/Paris"``).
        color : str, optional
            Matplotlib colour string; ``None`` uses the default cycle.
        marker : str, optional
            Matplotlib marker string; ``None`` uses no markers.
        linestyle : str, optional
            Matplotlib linestyle string; ``None`` uses the default.
        markevery : int, optional
            Draw a marker every *n* data points; ``None`` for every point.

        Raises
        ------
        RuntimeError
            If *x* or *y* is not a valid column name.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        logger.info(f"plotData: plotting {y} vs {x} from {self.FileName!r}")
        matplotlib.rcParams["text.usetex"] = True

        if x not in self.Keys + ["t", "timestamp"]:
            raise RuntimeError(
                f"{self.__class__.__name__}.plotData: no x={x} key (valid keys= {self.Keys})"
            )
        if y not in self.Keys:
            raise RuntimeError(
                f"{self.__class__.__name__}.plotData: no {y} key (valid keys: {self.Keys})"
            )

        ysymbol, yunit = self.getUnitKey(y)

        if x == "timestamp":
            x_values = series_utc_to_local_naive(
                self.Data["timestamp"].to_pandas(), time_zone
            ).to_numpy()
        else:
            x_values = self.Data[x].to_numpy()

        y_values = self.Data[y].to_numpy()

        plot_kwargs: dict = {"alpha": alpha}
        if color is not None:
            plot_kwargs["color"] = color
        if marker is not None:
            plot_kwargs["marker"] = marker
        if linestyle is not None:
            plot_kwargs["linestyle"] = linestyle
        if markevery is not None:
            plot_kwargs["markevery"] = markevery

        if normalize:
            ymax = abs(float(np.nanmax(y_values)))
            y_values = y_values / ymax
            plot_kwargs["label"] = f"{label or y} (norm with {ymax:.3e} {yunit:~P})"
        elif label is not None:
            plot_kwargs["label"] = label

        ax.plot(x_values, y_values, **plot_kwargs)

        if yunit is not None:
            logger.info(
                f"ysymbol={ysymbol}, yunit={yunit:~P}, labeling y-axis accordingly"
            )
            plt.ylabel(f"{ysymbol} [{yunit:~P}]")

        xsymbol, xunit = self.getUnitKey(x)
        if xunit is not None:
            logger.info(
                f"plotData: xsymbol={xsymbol}, xunit={xunit:~P}, labeling x-axis accordingly"
            )
            plt.xlabel(f"{xsymbol} [{xunit:~P}]")

    def stats(self, key: str | None = None) -> pl.DataFrame | None:
        """Print descriptive statistics for the dataset.

        Parameters
        ----------
        key : str, optional
            Restrict output to this column; ``None`` describes all columns
            (result is printed, not returned).

        Returns
        -------
        None
            Statistics are printed to stdout.

        Raises
        ------
        RuntimeError
            If *key* is given but not present in :attr:`Keys`.
        """
        from tabulate import tabulate

        logger.info("magnetdata.stats")
        if key is not None:
            if key in self.Keys:
                desc = self.Data[key].describe()
                logger.info(
                    tabulate(desc.rows(), headers=desc.columns, tablefmt="psql")
                )
            else:
                raise RuntimeError(f"{self.__class__.__name__}.stats: no {key} key")
        else:
            df = self.Data.describe()
            print(tabulate(df.rows(), headers=df.columns, tablefmt="psql"))
        return None

    # --- construction ------------------------------------------------------

    @classmethod
    def fromtxt(
        cls, name: str, defs_file: str | None = "pupitre-defs.json"
    ) -> PolarsMagnetData:
        """Create from a pupitre ``.txt`` file.

        Only the first data row is read at construction time so that
        :meth:`_validate_start_timestamp` can cross-check the filename
        timestamp. The full file is loaded lazily on first access, via
        :meth:`_ensure_data_loaded`.

        Parameters
        ----------
        name : str
            Path to the ``.txt`` file.
        defs_file : str, optional
            Path to a JSON field-definition file; defaults to
            ``"pupitre-defs.json"``.

        Returns
        -------
        PolarsMagnetData
            Fully initialised instance with lazy-loaded data.
        """
        from .readers.csv_readers import PupitreReader
        from .utils.validation import FileFormatError, check_pupitre_truncation

        if os.path.splitext(name)[-1] != ".txt":
            raise FileFormatError(f"{name}: expected .txt extension")
        reader = PupitreReader()
        reader.validate(name)
        stub = reader.read_stub_polars(name)
        if stub.height == 0:
            raise FileFormatError(f"{name}: no data rows found (header-only file)")
        Keys = stub.columns
        check_pupitre_truncation(name, Keys)
        return cls(name, {}, Keys, stub, defs_file=defs_file)
