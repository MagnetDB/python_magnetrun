"""CSV-format readers for pupitre, B-profile, Ensight, Feel++, and generic CSV."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..utils.files import _open_text_with_fallback

if TYPE_CHECKING:
    import polars as pl


class PupitreReader:
    """Reader for pupitre ``.txt`` whitespace-separated files.

    Attributes
    ----------
    sep : str
        Column separator regex (``r"\\s+"``).
    skip_rows : int
        Header rows to skip (``1`` — first line is a comment).
    on_bad_lines : str
        Behaviour on parse errors (``"warn"``).
    defs_file : str
        Default field-definition JSON file name.
    """

    sep: str = r"\s+"
    skip_rows: int = 1
    on_bad_lines: str = "warn"
    defs_file: str = "pupitre-defs.json"

    def read(self, path: Path) -> pd.DataFrame:
        """Read the full file.

        Parameters
        ----------
        path : Path
            Pupitre ``.txt`` file.

        Returns
        -------
        pd.DataFrame
            Parsed data with all rows.
        """
        with _open_text_with_fallback(path) as f:
            return pd.read_csv(
                f,
                sep=self.sep,
                skiprows=self.skip_rows,
                on_bad_lines=self.on_bad_lines,
            )

    def read_stub(self, path: Path) -> pd.DataFrame:
        """Read first data row only — used to infer column names cheaply.

        Parameters
        ----------
        path : Path
            Pupitre ``.txt`` file.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame used for key discovery.
        """
        with _open_text_with_fallback(path) as f:
            return pd.read_csv(
                f,
                sep=self.sep,
                skiprows=self.skip_rows,
                on_bad_lines=self.on_bad_lines,
                nrows=1,
            )

    def read_kwargs(self) -> dict:
        """Return ``pd.read_csv`` kwargs stored on the container for lazy loading.

        Returns
        -------
        dict
            Keyword arguments compatible with :func:`pandas.read_csv`.
        """
        return {
            "sep": self.sep,
            "skiprows": self.skip_rows,
            "on_bad_lines": self.on_bad_lines,
        }

    def _parse_header(self, path: Path) -> list[str]:
        """Parse the column-name row with the same whitespace regex as ``read()``.

        Needed because some real pupitre files have doubled tabs in the
        header row that don't match the single-tab data rows below it —
        Polars' literal-separator ``read_csv`` cannot parse that correctly,
        but the ``sep`` regex used by ``read()`` collapses it as intended.

        Parameters
        ----------
        path : Path
            Pupitre ``.txt`` file.

        Returns
        -------
        list of str
            Column names.
        """
        with _open_text_with_fallback(path) as f:
            for _ in range(self.skip_rows):
                f.readline()
            header_line = f.readline()
        return re.split(self.sep, header_line.strip())

    def _read_polars_impl(self, path: Path, n_rows: int | None = None) -> pl.DataFrame:
        """Read *path* as a Polars DataFrame, optionally limited to *n_rows*.

        The header row is parsed separately (see ``_parse_header``) and the
        data body is read without Polars' own header inference, since Polars
        only supports a single literal separator character, not a regex.
        Some real files end each data row with a trailing tab — one extra
        field beyond the real column count, silently absorbed by pandas'
        whitespace-regex split — which is dropped here after confirming it
        is all-null; files without that trailing tab are left as-is. Header-
        only files (no data rows) return a correctly-shaped empty DataFrame
        instead of raising.

        Unlike ``read()``, this does not fall back to Latin-1 on decode
        errors for the data body — Polars' ``read_csv`` only supports UTF-8
        (the header line itself still goes through the same encoding
        fallback as ``read()``, via ``_parse_header``).

        Parameters
        ----------
        path : Path
            Pupitre ``.txt`` file.
        n_rows : int, optional
            Maximum number of data rows to read.

        Returns
        -------
        pl.DataFrame
            Parsed data.
        """
        import polars as pl

        columns = self._parse_header(path)
        try:
            df = pl.read_csv(
                path,
                separator="\t",
                skip_rows=self.skip_rows + 1,
                has_header=False,
                n_rows=n_rows,
            )
        except pl.exceptions.NoDataError:
            return pl.DataFrame({col: [] for col in columns})

        n_extra = df.width - len(columns)
        if n_extra == 0:
            df.columns = columns
            return df
        if n_extra != 1:
            raise AssertionError(
                f"expected at most one trailing artifact column, got {n_extra} "
                f"(header has {len(columns)} names, body has {df.width} columns)"
            )
        trailing = df.columns[-1]
        if not df[trailing].is_null().all():
            raise AssertionError(
                f"expected trailing column {trailing!r} to be all-null "
                "(format assumption changed) — refusing to silently drop it"
            )
        df = df.drop(trailing)
        df.columns = columns
        return df

    def read_polars(self, path: Path) -> pl.DataFrame:
        """Read the full file as a Polars DataFrame.

        Parameters
        ----------
        path : Path
            Pupitre ``.txt`` file.

        Returns
        -------
        pl.DataFrame
            Parsed data with all rows.
        """
        return self._read_polars_impl(path)

    def read_stub_polars(self, path: Path) -> pl.DataFrame:
        """Read first data row only as a Polars DataFrame.

        Parameters
        ----------
        path : Path
            Pupitre ``.txt`` file.

        Returns
        -------
        pl.DataFrame
            Single-row (or empty, for header-only files) DataFrame used for
            key discovery.
        """
        return self._read_polars_impl(path, n_rows=1)

    def validate(self, path: Path) -> bool:
        """Validate a pupitre ``.txt`` file.

        Parameters
        ----------
        path : Path
            File to validate.

        Returns
        -------
        bool
            Always ``True`` on success; raises on failure.
        """
        from ..utils.validation import validate_txt_format

        validate_txt_format(str(path))
        return True


class BProfileReader:
    """Reader for B-profile whitespace-separated files (Index/Position/Profile).

    Attributes
    ----------
    sep : str
        Column separator regex (``r"\\s+"``).
    engine : str
        pandas CSV engine (``"python"``).
    skip_rows : int
        Header rows to skip (``0``).
    expected_cols : list[str]
        Expected column names used for validation.
    defs_file : None
        No default defs file for this format.
    """

    sep: str = r"\s+"
    engine: str = "python"
    skip_rows: int = 0
    expected_cols: list[str] = ["Index", "Position", "Profile"]
    defs_file: None = None

    def read(self, path: Path) -> pd.DataFrame:
        """Read a B-profile file.

        Parameters
        ----------
        path : Path
            B-profile whitespace-separated file.

        Returns
        -------
        pd.DataFrame
            Parsed data.
        """
        with open(path) as f:
            return pd.read_csv(
                f,
                sep=self.sep,
                engine=self.engine,
                skiprows=self.skip_rows,
            )

    def read_kwargs(self) -> dict:
        """Return ``pd.read_csv`` kwargs for lazy loading.

        Returns
        -------
        dict
            Keyword arguments compatible with :func:`pandas.read_csv`.
        """
        return {"sep": self.sep, "engine": self.engine, "skiprows": self.skip_rows}

    def validate(self, path: Path) -> bool:
        """Validate a B-profile CSV file.

        Parameters
        ----------
        path : Path
            File to validate.

        Returns
        -------
        bool
            Always ``True`` on success; raises on failure.
        """
        from ..utils.validation import validate_csv_format

        validate_csv_format(str(path))
        return True


class EnsightReader:
    """Reader for Ensight CSV files (two-row header, comma-separated).

    Attributes
    ----------
    sep : str
        Column separator (``","``).
    engine : str
        pandas CSV engine (``"python"``).
    skip_rows : int
        Ensight header rows to skip (``2``).
    defs_file : None
        No default defs file for this format.
    """

    sep: str = ","
    engine: str = "python"
    skip_rows: int = 2
    defs_file: None = None

    def read(self, path: Path) -> pd.DataFrame:
        """Read an Ensight CSV file.

        Parameters
        ----------
        path : Path
            Ensight ``.csv`` file.

        Returns
        -------
        pd.DataFrame
            Parsed data.
        """
        with open(path) as f:
            return pd.read_csv(
                f,
                sep=self.sep,
                engine=self.engine,
                skiprows=self.skip_rows,
            )

    def read_kwargs(self) -> dict:
        """Return ``pd.read_csv`` kwargs for lazy loading.

        Returns
        -------
        dict
            Keyword arguments compatible with :func:`pandas.read_csv`.
        """
        return {"sep": self.sep, "engine": self.engine, "skiprows": self.skip_rows}

    def validate(self, path: Path) -> bool:
        """Validate an Ensight CSV file (existence check only).

        Parameters
        ----------
        path : Path
            File to validate.

        Returns
        -------
        bool
            Always ``True`` on success; raises on failure.
        """
        from ..utils.validation import validate_file_exists

        validate_file_exists(str(path))
        return True


class FeelppReader:
    """Reader for Feel++ simulation CSV files (configurable header skip).

    Attributes
    ----------
    sep : str
        Column separator (``","``).
    engine : str
        pandas CSV engine (``"python"``).
    skip_rows : int
        Header rows to skip (default ``0``, configurable via constructor).
    defs_file : str
        Default field-definition JSON file name.
    """

    sep: str = ","
    engine: str = "python"
    defs_file: str = "feelpp-defs.json"

    def __init__(self, skip_rows: int = 0) -> None:
        """Initialise with a configurable number of header rows to skip.

        Parameters
        ----------
        skip_rows : int, optional
            Number of header rows to skip (default ``0``).
        """
        self.skip_rows: int = skip_rows

    def read(self, path: Path) -> pd.DataFrame:
        """Read a Feel++ CSV file.

        Parameters
        ----------
        path : Path
            Feel++ ``.csv`` file.

        Returns
        -------
        pd.DataFrame
            Parsed data.
        """
        with open(path) as f:
            return pd.read_csv(
                f,
                sep=self.sep,
                engine=self.engine,
                skiprows=self.skip_rows,
            )

    def read_kwargs(self) -> dict:
        """Return ``pd.read_csv`` kwargs for lazy loading.

        Returns
        -------
        dict
            Keyword arguments compatible with :func:`pandas.read_csv`.
        """
        return {"sep": self.sep, "engine": self.engine, "skiprows": self.skip_rows}

    def validate(self, path: Path) -> bool:
        """Validate a Feel++ CSV file.

        Parameters
        ----------
        path : Path
            File to validate.

        Returns
        -------
        bool
            Always ``True`` on success; raises on failure.
        """
        from ..utils.validation import validate_csv_format

        validate_csv_format(str(path))
        return True


class CsvReader:
    """Generic comma-separated reader (no header skip).

    Attributes
    ----------
    sep : str
        Column separator (``","``).
    engine : str
        pandas CSV engine (``"python"``).
    skip_rows : int
        Header rows to skip (``0``).
    on_bad_lines : str
        Behaviour on parse errors (``"warn"``).
    defs_file : None
        No default defs file for this format.
    """

    sep: str = ","
    engine: str = "python"
    skip_rows: int = 0
    on_bad_lines: str = "warn"
    defs_file: None = None

    def read(self, path: Path) -> pd.DataFrame:
        """Read a generic CSV file.

        Parameters
        ----------
        path : Path
            CSV file.

        Returns
        -------
        pd.DataFrame
            Parsed data.
        """
        with _open_text_with_fallback(path) as f:
            return pd.read_csv(
                f,
                sep=self.sep,
                engine=self.engine,
                skiprows=self.skip_rows,
                on_bad_lines=self.on_bad_lines,
            )

    def read_kwargs(self) -> dict:
        """Return ``pd.read_csv`` kwargs for lazy loading.

        Returns
        -------
        dict
            Keyword arguments compatible with :func:`pandas.read_csv`.
        """
        return {
            "sep": self.sep,
            "engine": self.engine,
            "skiprows": self.skip_rows,
            "on_bad_lines": self.on_bad_lines,
        }

    def validate(self, path: Path) -> bool:
        """Validate a generic CSV file.

        Parameters
        ----------
        path : Path
            File to validate.

        Returns
        -------
        bool
            Always ``True`` on success; raises on failure.
        """
        from ..utils.validation import validate_csv_format

        validate_csv_format(str(path))
        return True
