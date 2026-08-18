"""Reader registry — maps DataType → reader class and container class.

Use :func:`detect_type` to infer the format from a file path, then look up
the appropriate reader in :data:`READERS` and the container in
:data:`CONTAINERS`.

Example
-------
::

    from python_magnetrun.readers.registry import READERS, CONTAINERS, detect_type
    from pathlib import Path

    path = Path("run_2025-01-06.txt")
    dtype = detect_type(path)
    reader = READERS[dtype]()
    container_cls = CONTAINERS[dtype]

"""

from __future__ import annotations

from pathlib import Path

from ..magnetdata_base import DataType


def detect_type(path: Path, fmt: str | None = None) -> DataType:
    """Infer :class:`~python_magnetrun.magnetdata_base.DataType` from *path*.

    Parameters
    ----------
    path : Path
        File (or directory) to classify.
    fmt : str, optional
        Explicit format override (case-insensitive ``DataType`` member name,
        e.g. ``"tdms"``, ``"pupitre"``).  When provided, extension detection
        is skipped.

    Returns
    -------
    DataType
        Detected data type.

    Raises
    ------
    ValueError
        If *fmt* is not a valid ``DataType`` name, or if the extension is
        not recognised and *fmt* is ``None``.
    """
    if fmt is not None:
        try:
            return DataType[fmt.upper()]
        except KeyError:
            valid = [m.name for m in DataType]
            raise ValueError(
                f"detect_type: unknown format {fmt!r}; valid values are {valid}"
            ) from None

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tdms":
        return DataType.TDMS
    if suffix == ".txt":
        return DataType.PUPITRE
    if suffix == ".csv":
        return DataType.PUPITRE
    if path.is_dir():
        return DataType.HYBRID
    raise ValueError(
        f"detect_type: cannot detect DataType for {path!r} "
        "(pass fmt= to override)"
    )


def _build_readers() -> dict:
    """Build the READERS mapping lazily to avoid circular imports at module load."""
    from .csv_readers import EnsightReader, PupitreReader
    from .hts_reader import HtsReader
    from .hybrid_reader import HybridReader
    from .tdms_reader import TdmsReader

    return {
        DataType.PUPITRE: PupitreReader,
        DataType.TDMS: TdmsReader,
        DataType.ENSIGHT: EnsightReader,
        DataType.HYBRID: HybridReader,
        DataType.HTS: HtsReader,
    }


def _build_containers() -> dict:
    """Build the CONTAINERS mapping lazily."""
    from ..hybrid.hybrid_data import HybridData
    from ..magnetdata_pandas import (
        EnsightMagnetData,
        PandasMagnetData,
    )
    from ..magnetdata_tdms import TdmsMagnetData

    return {
        DataType.PUPITRE: PandasMagnetData,
        DataType.TDMS: TdmsMagnetData,
        DataType.ENSIGHT: EnsightMagnetData,
        DataType.HYBRID: HybridData,
        DataType.HTS: PandasMagnetData,
    }


# Lazy proxies — evaluated on first access to avoid circular imports.
class _LazyDict:
    """Dict-like wrapper whose contents are built on first access."""

    def __init__(self, factory):
        self._factory = factory
        self._cache: dict | None = None

    def _ensure(self) -> dict:
        if self._cache is None:
            self._cache = self._factory()
        return self._cache

    def __getitem__(self, key):
        return self._ensure()[key]

    def __contains__(self, key):
        return key in self._ensure()

    def keys(self):
        return self._ensure().keys()

    def items(self):
        return self._ensure().items()

    def values(self):
        return self._ensure().values()


READERS: _LazyDict = _LazyDict(_build_readers)
"""Mapping of :class:`~python_magnetrun.magnetdata_base.DataType` → reader class."""

CONTAINERS: _LazyDict = _LazyDict(_build_containers)
"""Mapping of :class:`~python_magnetrun.magnetdata_base.DataType` → container class."""
