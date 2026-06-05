"""CSV-format readers for pupitre, B-profile, Ensight, Feel++, and generic CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _open_text(path: Path):  # type: ignore[return]
    """Open *path* with encoding fallback (UTF-8 → latin-1)."""
    from ..utils.files import _open_text_with_fallback

    return _open_text_with_fallback(str(path))


class PupitreReader:
    """Reader for pupitre ``.txt`` whitespace-separated files.

    Attributes
    ----------
    sep : str
        Column separator regex (``r"\\s+"``).
    engine : str
        pandas CSV engine (``"python"``).
    skip_rows : int
        Header rows to skip (``1`` — first line is a comment).
    on_bad_lines : str
        Behaviour on parse errors (``"warn"``).
    defs_file : str
        Default field-definition JSON file name.
    """

    sep: str = r"\s+"
    engine: str = "python"
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
        with _open_text(path) as f:
            return pd.read_csv(
                f,
                sep=self.sep,
                engine=self.engine,
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
        with _open_text(path) as f:
            return pd.read_csv(
                f,
                sep=self.sep,
                engine=self.engine,
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
            "engine": self.engine,
            "skiprows": self.skip_rows,
            "on_bad_lines": self.on_bad_lines,
        }

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
        with _open_text(path) as f:
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
