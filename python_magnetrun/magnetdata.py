"""magnetdata — factory entry point and re-exports.

The concrete implementations live in:
- :mod:`python_magnetrun.magnetdata_base`   — ``MagnetDataBase`` ABC
- :mod:`python_magnetrun.magnetdata_pandas` — ``PandasMagnetData`` and subclasses
- :mod:`python_magnetrun.magnetdata_tdms`   — ``TdmsMagnetData``

Use :func:`load_magnetdata` to load a file by extension.  Import the concrete
classes directly for construction or isinstance checks.
"""

import logging
import os
import time

from .magnetdata_base import DataType, MagnetDataBase
from .magnetdata_pandas import (
    BProfileMagnetData,
    EnsightMagnetData,
    FeelppMagnetData,
    PandasMagnetData,
)
from .magnetdata_tdms import TdmsMagnetData
from .utils.validation import FileFormatError

logger = logging.getLogger(__name__)

__all__ = [
    "MagnetDataBase",
    "DataType",
    "PandasMagnetData",
    "EnsightMagnetData",
    "BProfileMagnetData",
    "FeelppMagnetData",
    "TdmsMagnetData",
    "FileFormatError",
    "load_magnetdata",
]


def load_magnetdata(
    filename: str,
    defs_file: str | None = None,
    fmt: str | None = None,
) -> MagnetDataBase:
    """Load a magnet data file and return the appropriate MagnetDataBase subclass.

    Dispatches via :func:`~python_magnetrun.readers.registry.detect_type`:

    - ``.tdms`` → :class:`TdmsMagnetData`
    - ``.txt``  → :class:`PandasMagnetData`
    - ``.csv``  → :class:`PandasMagnetData`

    Parameters
    ----------
    filename : str
        Path to the data file.
    defs_file : str, optional
        Path to a field definitions JSON file.
    fmt : str, optional
        Explicit format override (``DataType`` member name, e.g.
        ``"tdms"``); when provided, extension detection is skipped.

    Returns
    -------
    MagnetDataBase
        The loaded data object.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    from .readers.registry import DataType, detect_type

    data_type = detect_type(os.fspath(filename), fmt=fmt)

    if data_type == DataType.TDMS:
        return _fromtdms(filename, defs_file=defs_file or "pigbrother-defs.json")
    elif data_type == DataType.PUPITRE:
        ext = os.path.splitext(filename)[-1].lower()
        if ext == ".txt":
            return PandasMagnetData.fromtxt(
                filename, defs_file=defs_file or "pupitre-defs.json"
            )
        return PandasMagnetData.fromcsv(filename, defs_file=defs_file)
    else:
        raise ValueError(
            f"load_magnetdata: unsupported format {data_type.name!r} for {filename!r}"
        )


def _fromtdms(
    name: str, defs_file: str | None = "pigbrother-defs.json"
) -> TdmsMagnetData:
    """Load a pigbrother TDMS file and return a :class:`TdmsMagnetData`.

    This function contains the TDMS-specific loading logic previously on
    ``MagnetData.fromtdms``.

    Parameters
    ----------
    name : str
        Filename with a ``.tdms`` extension.
    defs_file : str, optional
        Path to a field definitions file; defaults to the bundled
        ``pigbrother-defs.json``.

    Returns
    -------
    TdmsMagnetData
        Loaded TDMS data instance.

    Raises
    ------
    FileNotFoundError
        If *name* does not exist.
    RuntimeError
        If file extension is not ``.tdms`` or required group is missing.
    """
    from nptdms import TdmsFile

    from .readers.tdms_reader import TdmsReader

    logger.debug(f"load_magnetdata/_fromtdms: {name}")

    Keys: list[str] = []
    Groups: dict = {}
    # _tdms_groups maps normalised group name → TdmsGroup for lazy data loading
    _tdms_groups: dict = {}

    if not os.path.exists(name):
        raise FileNotFoundError(f"_fromtdms: file not found: {name}")
    f_extension = os.path.splitext(name)[-1]
    if f_extension != ".tdms":
        raise RuntimeError(f"_fromtdms: expect a tdms filename - got {name}")

    reader = TdmsReader()
    reader.validate(name)
    t_offset: float = reader.t_offset_for(name)

    # Keep handle open — data arrays are read lazily via _ensure_group_loaded()
    t0 = time.perf_counter()
    rawData = TdmsFile.open(name)
    elapsed = time.perf_counter() - t0
    mib = os.path.getsize(name) / 1024**2
    logger.debug(f"tdms.io: open file={name} size={mib:.1f}MiB time={elapsed:.3f}s")
    for group in rawData.groups():
        gname = group.name.replace(" ", "_")
        gname = gname.replace("_et_Ref.", "")
        if gname != group.name:
            logger.warning(
                f"fromtdms: group name rewritten {group.name!r} -> {gname!r} (old TDMS format)"
            )
        Groups[gname] = {}
        if gname != "Infos":
            _tdms_groups[gname] = group
            for channel in group.channels():
                cname = channel.name.replace(" ", "_")
                Keys.append(f"{gname}/{cname}")
                Groups[gname][cname] = channel.properties
                logger.debug(
                    f"channel {gname}/{cname} properties: {Groups[gname][cname]}"
                )
                if "wf_start_offset" in Groups[gname][cname]:
                    logger.debug(
                        f"update wf_start_offset for {gname}/{cname} - from: "
                        f"{Groups[gname][cname]['wf_start_offset']}"
                        f" to: {t_offset}"
                    )
                    Groups[gname][cname]["wf_start_offset"] = t_offset
        else:
            Groups[gname] = group

    if reader.required_group not in Groups:
        raise RuntimeError(
            f"_fromtdms: {reader.required_group} group not found in {name}"
        )

    mdata = TdmsMagnetData(
        name,
        Groups,
        Keys,
        Data={},
        defs_file=defs_file,
        _tdms_file=rawData,
        _tdms_groups=_tdms_groups,
    )
    return mdata
