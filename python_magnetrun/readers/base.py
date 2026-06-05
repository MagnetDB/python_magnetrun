"""Reader protocol — interface every format reader must satisfy."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Reader(Protocol):
    """Minimal protocol for format readers.

    A reader is a stateless (or configuration-only) object whose sole
    responsibility is I/O: reading bytes from disk and returning a tidy
    :class:`~pandas.DataFrame` (or a dict of DataFrames for multi-group
    formats like TDMS).  No data manipulation happens here.

    Implementations live in :mod:`python_magnetrun.readers`.
    """

    def read(self, path: Path) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Read *path* and return a DataFrame (or dict thereof).

        Parameters
        ----------
        path : Path
            Source file or directory to read.

        Returns
        -------
        pd.DataFrame or dict[str, pd.DataFrame]
            Parsed data.
        """
        ...

    def validate(self, path: Path) -> bool:
        """Return ``True`` when *path* is a valid file for this reader.

        Raises format-specific exceptions on failure rather than returning
        ``False`` so that callers get actionable error messages.

        Parameters
        ----------
        path : Path
            File to validate.

        Returns
        -------
        bool
            Always ``True`` on success.
        """
        ...
