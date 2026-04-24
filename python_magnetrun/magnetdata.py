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
) -> MagnetDataBase:
    """Load a magnet data file and return the appropriate MagnetDataBase subclass.

    Dispatches on file extension:

    - ``.tdms`` → :class:`TdmsMagnetData`
    - ``.txt``  → :class:`PandasMagnetData`
    - ``.csv``  → :class:`PandasMagnetData`

    :param filename: path to the data file
    :param defs_file: optional path to a field definitions JSON file
    :return: the loaded data object
    :raises ValueError: if the file extension is not recognised
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".tdms":
        if defs_file is not None:
            return _fromtdms(filename, defs_file=defs_file)
        return _fromtdms(filename)
    elif ext == ".txt":
        return PandasMagnetData.fromtxt(
            filename, defs_file=defs_file or "pupitre-defs.json"
        )
    elif ext == ".csv":
        return PandasMagnetData.fromcsv(filename, defs_file=defs_file)
    else:
        raise ValueError(f"load_magnetdata: unsupported file extension {ext!r}")


def _fromtdms(
    name: str, defs_file: str | None = "pigbrother-defs.json"
) -> TdmsMagnetData:
    """Load a pigbrother TDMS file and return a :class:`TdmsMagnetData`.

    This function contains the TDMS-specific loading logic previously on
    ``MagnetData.fromtdms``.

    :param name: filename with a .tdms extension
    :param defs_file: path to a field definitions file; defaults to the
        bundled ``pigbrother-defs.json``
    :raises FileNotFoundError: if *name* does not exist
    :raises RuntimeError: if file extension is not .tdms or required group missing
    :return: TdmsMagnetData instance
    """
    import pandas as pd
    from nptdms import TdmsFile

    from .utils.validation import validate_tdms_format

    logger.debug(f"load_magnetdata/_fromtdms: {name}")

    Keys: list[str] = []
    Groups: dict = {}
    Data: dict[str, pd.DataFrame] = {}

    if not os.path.exists(name):
        raise FileNotFoundError(f"_fromtdms: file not found: {name}")
    f_extension = os.path.splitext(name)[-1]
    if f_extension != ".tdms":
        raise RuntimeError(f"_fromtdms: expect a tdms filename - got {name}")
    validate_tdms_format(name)

    t_offset: float = 0.0
    if "Overview" in name:
        t_offset = 1 / 2.0
    elif "Archive" in name:
        t_offset = (1 / 120.0) / 2.0

    rawData = TdmsFile.open(name)
    for group in rawData.groups():
        gname = group.name.replace(" ", "_")
        gname = gname.replace("_et_Ref.", "")
        if gname != group.name:
            logger.warning(
                "fromtdms: group name rewritten %r -> %r (old TDMS format)",
                group.name,
                gname,
            )
        Groups[gname] = {}
        if gname != "Infos":
            Data[gname] = {}
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
            Data[gname] = group.as_dataframe(
                time_index=False, absolute_time=False, scaled_data=True
            )
            Data[gname].rename(
                columns={col: col.replace(" ", "_") for col in Data[gname].columns},
                inplace=True,
            )
            channel = list(Groups[gname].keys())[0]
            if len(Data[gname]) != Groups[gname][channel]["wf_samples"]:
                logger.warning(
                    f"group '{gname}': loaded {len(Data[gname])} rows but wf_samples={Groups[gname][channel]['wf_samples']}"
                )
                logging.warning(
                    f"update wf_samples for {gname} to match loaded data length: {len(Data[gname])}"
                )
                for channel in Groups[gname]:
                    Groups[gname][channel]["wf_samples"] = len(Data[gname])
        else:
            Groups[gname] = group

    if "Courants_Alimentations" not in Data:
        raise RuntimeError(
            f"_fromtdms: Courants_Alimentations group not found in {name}"
        )

    mdata = TdmsMagnetData(name, Groups, Keys, Data, defs_file=defs_file)
    return mdata
