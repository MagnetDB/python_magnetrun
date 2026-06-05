"""HTS reader — semicolon-separated files with units embedded in column headers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


class HtsReader:
    """Reader for HTS files: ``;``-separated, column headers carry units.

    Header line format: ``"Temps [s];I_H1 [A];U_H1 [V]"``
    The reader strips the bracketed unit suffix from each column name and
    exposes the unit map via :meth:`extracted_units`.

    Attributes
    ----------
    sep : str
        Column separator (``";"``).
    skip_rows : int
        Header rows to skip after the unit-bearing header (``0``).
    header_units : bool
        When ``True`` the reader parses ``"Col [unit]"`` from the header row.
    defs_file : str
        Default field-definition JSON file name (shared with Feel++).
    """

    sep: str = ";"
    skip_rows: int = 0
    header_units: bool = True
    defs_file: str = "feelpp-defs.json"

    # Matches "Col [unit]" or "Col [unit/unit]"
    _UNIT_RE: re.Pattern[str] = re.compile(r"^(.+?)\s*\[([^\]]*)\]\s*$")

    def _parse_header(self, path: Path) -> tuple[list[str], dict[str, str]]:
        """Parse the first line to extract clean column names and units.

        Parameters
        ----------
        path : Path
            HTS file to inspect.

        Returns
        -------
        tuple[list[str], dict[str, str]]
            ``(clean_names, units_map)`` where *clean_names* are column names
            with the ``[unit]`` suffix removed and *units_map* maps each clean
            name to its unit string (empty string when absent).
        """
        with open(path, encoding="utf-8", errors="replace") as fh:
            header_line = fh.readline().rstrip("\n")

        raw_cols = header_line.split(self.sep)
        clean_names: list[str] = []
        units_map: dict[str, str] = {}
        for col in raw_cols:
            col = col.strip()
            m = self._UNIT_RE.match(col)
            if m:
                name, unit = m.group(1).strip(), m.group(2).strip()
            else:
                name, unit = col, ""
            clean_names.append(name)
            units_map[name] = unit
        return clean_names, units_map

    def read(self, path: Path) -> pd.DataFrame:
        """Read an HTS file and return a DataFrame with clean column names.

        The ``[unit]`` suffix is stripped from every column name so that
        column names are bare identifiers (e.g. ``"Temps"``, ``"I_H1"``).

        Parameters
        ----------
        path : Path
            HTS ``;``-separated file.

        Returns
        -------
        pd.DataFrame
            Parsed data with normalised column names.
        """
        clean_names, _ = self._parse_header(path)
        df = pd.read_csv(
            path,
            sep=self.sep,
            skiprows=self.skip_rows,
            header=0,
            names=clean_names,
            encoding="utf-8",
            encoding_errors="replace",
        )
        return df

    def extracted_units(self, path: Path) -> dict[str, str]:
        """Return ``{column_name: unit_string}`` parsed from the header.

        Parameters
        ----------
        path : Path
            HTS file to inspect.

        Returns
        -------
        dict[str, str]
            Mapping of clean column name → unit string (e.g. ``"A"``, ``"s"``).
            Columns without a unit annotation map to an empty string.
        """
        _, units_map = self._parse_header(path)
        return units_map

    def validate(self, path: Path) -> bool:
        """Validate that *path* exists and appears to be an HTS file.

        Checks that the file exists and that the first line contains at least
        one ``;`` separator.

        Parameters
        ----------
        path : Path
            File to validate.

        Returns
        -------
        bool
            Always ``True`` on success; raises on failure.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the first line does not look like an HTS file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HtsReader.validate: file not found: {path}")
        with open(path, encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        if self.sep not in first_line:
            raise ValueError(
                f"HtsReader.validate: {path} does not appear to be an HTS file "
                f"(no {self.sep!r} separator in first line)"
            )
        return True
