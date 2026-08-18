"""magnettools output file reader.

The exact magnettools file format (delimiter, column names, header structure,
time axis) is not yet confirmed.  This stub attempts generic CSV loading via
``load_magnetdata()`` and should be replaced once the format is known.
"""

from __future__ import annotations


def load_magnettools(filename: str):
    """Load a magnettools output file.

    Returns a :class:`~python_magnetrun.magnetdata_pandas.PandasMagnetData`
    instance.  The exact format (CSV delimiter, column names, header lines,
    time axis) must be confirmed with the user before this stub can be
    finalised.
    """
    from ..magnetdata import load_magnetdata

    return load_magnetdata(filename)
