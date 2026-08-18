"""readers — format-specific I/O classes for python_magnetrun.

Each reader holds only the I/O configuration (separator, skip rows, encoding,
validation) for one file format.  No data manipulation happens inside readers.

Public interface
----------------
:class:`Reader`
    Runtime-checkable protocol that all readers satisfy.
:data:`READERS`
    ``{DataType → reader class}`` registry; lazily populated.
:data:`CONTAINERS`
    ``{DataType → container class}`` registry; lazily populated.
:func:`detect_type`
    Infer :class:`~python_magnetrun.magnetdata_base.DataType` from a path.

Reader classes
--------------
:class:`~python_magnetrun.readers.csv_readers.PupitreReader`
    Pupitre ``.txt`` whitespace-separated files.
:class:`~python_magnetrun.readers.csv_readers.BProfileReader`
    B-profile whitespace-separated files.
:class:`~python_magnetrun.readers.csv_readers.EnsightReader`
    Ensight CSV files (two-row header).
:class:`~python_magnetrun.readers.csv_readers.FeelppReader`
    Feel++ simulation CSV files (configurable header skip).
:class:`~python_magnetrun.readers.csv_readers.CsvReader`
    Generic comma-separated files.
:class:`~python_magnetrun.readers.tdms_reader.TdmsReader`
    Pigbrother TDMS files.
:class:`~python_magnetrun.readers.hts_reader.HtsReader`
    HTS files — ``;``-separated with units in column headers.
:class:`~python_magnetrun.readers.hybrid_reader.HybridReader`
    Hybrid FEPC acquisition data (composite kHz/rms/trigger).
"""

from .base import Reader
from .csv_readers import (
    BProfileReader,
    CsvReader,
    EnsightReader,
    FeelppReader,
    PupitreReader,
)
from .hts_reader import HtsReader
from .hybrid_reader import HybridReader
from .registry import CONTAINERS, READERS, detect_type
from .tdms_reader import TdmsReader

__all__ = [
    "Reader",
    "PupitreReader",
    "BProfileReader",
    "EnsightReader",
    "FeelppReader",
    "CsvReader",
    "TdmsReader",
    "HtsReader",
    "HybridReader",
    "READERS",
    "CONTAINERS",
    "detect_type",
]
