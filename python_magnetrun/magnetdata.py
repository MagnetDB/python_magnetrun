"""MagnetData — factory class and backward-compatible entry point.

The concrete implementations live in:
- :mod:`python_magnetrun.magnetdata_base`   — ``MagnetDataBase`` ABC
- :mod:`python_magnetrun.magnetdata_pandas` — ``PandasMagnetData`` and subclasses
- :mod:`python_magnetrun.magnetdata_tdms`   — ``TdmsMagnetData``

``MagnetData`` is kept for backward compatibility: external code that does
``from .magnetdata import MagnetData`` and calls ``MagnetData(...)`` or any
``MagnetData.from*`` classmethod continues to work unchanged.

``MagnetDataBase`` is also re-exported from this module so callers can
import it from the same place.
"""

import logging
import os

import pandas as pd

from .magnetdata_base import DataType, MagnetDataBase
from .magnetdata_pandas import (
    BProfileMagnetData,
    EnsightMagnetData,
    FeelppMagnetData,
    PandasMagnetData,
)
from .magnetdata_tdms import TdmsMagnetData
from .utils.validation import (
    FileFormatError,
    validate_csv_format,
    validate_file_exists,
    validate_tdms_format,
    validate_txt_format,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MagnetData",
    "MagnetDataBase",
    "DataType",
    "PandasMagnetData",
    "EnsightMagnetData",
    "BProfileMagnetData",
    "FeelppMagnetData",
    "TdmsMagnetData",
    "FileFormatError",
]


class MagnetData(PandasMagnetData):
    """Backward-compatible magnet data container and factory.

    Extends :class:`PandasMagnetData` so that direct construction
    ``MagnetData(filename, Groups, Keys, Type, Data)`` continues to work.
    The ``Type`` parameter is stored via a property; all method logic is
    inherited from :class:`PandasMagnetData`.

    The ``from*`` classmethods return the appropriate :class:`MagnetDataBase`
    subclass (e.g. :class:`PandasMagnetData`, :class:`TdmsMagnetData`).

    FileName: name of the input file
    Data: pandas dataframe
    Keys:
    Groups:
    Type: 0 for Pandas data (default), 1 for Tdms, 2 for Ensight
    """

    def __init__(
        self,
        filename: str,
        Groups: dict,
        Keys: list[str],
        Type: int | DataType = DataType.PUPITRE,
        Data: pd.DataFrame | dict | None = None,
        defs_file: str | None = None,
    ) -> None:
        """Default constructor — accepts the legacy 5-argument signature."""
        self._type: DataType = DataType(Type)
        # Call MagnetDataBase.__init__ directly to avoid PandasMagnetData asserting
        # that Data is always a DataFrame (Type=1/2 passes dict or empty df)
        MagnetDataBase.__init__(self, filename, Groups, Keys, Data, defs_file)
        self.units: dict = {}

    @property
    def Type(self) -> DataType:
        """Data-type discriminator."""
        return self._type

    # ------------------------------------------------------------------
    # Factory classmethods — return the appropriate subclass
    # ------------------------------------------------------------------

    @classmethod
    def fromtdms(cls, name: str, defs_file: str | None = None) -> "TdmsMagnetData":
        """Create from a pigbrother TDMS file.

        :param name: filename with a .tdms extension
        :param defs_file: optional path to a pigbrother-defs.json field definitions file
        :raises FileNotFoundError: if *name* does not exist
        :raises RuntimeError: if file extension is not .tdms or required group missing
        :return: TdmsMagnetData instance
        """
        from nptdms import TdmsFile

        logger.debug(f"magnetdata.fromtdms: {name}")

        Keys: list[str] = []
        Groups: dict = {}
        Data: dict[str, pd.DataFrame] = {}
        if not os.path.exists(name):
            raise FileNotFoundError(f"fromtdms: file not found: {name}")
        f_extension = os.path.splitext(name)[-1]

        if f_extension != ".tdms":
            raise RuntimeError(f"fromtdms: expect a tdms filename - got {name}")
        validate_tdms_format(name)

        # add t_offset to account for tdms downsampled data
        t_offset: float = 0.0
        if "Overview" in name:
            t_offset = (1) / 2.0
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
                    if "wf_start_offset" in Groups[gname][cname]:
                        logger.debug(
                            f"update wf_start_offset for {gname}/{cname} - original value: {Groups[gname][cname]['wf_start_offset']}"
                        )
                        Groups[gname][cname]["wf_start_offset"] = t_offset

                Data[gname] = group.as_dataframe(
                    time_index=False,
                    absolute_time=False,
                    scaled_data=True,
                )

                Data[gname].rename(
                    columns={col: col.replace(" ", "_") for col in Data[gname].columns},
                    inplace=True,
                )

            else:
                Groups[gname] = group

        if "Courants_Alimentations" not in Data:
            raise RuntimeError(
                f"magnetdata/fromtdms: Courants_Alimentations group not found in {name}"
            )

        # Add reference for GR1, GR2
        if "Référence_A1" in Data["Courants_Alimentations"]:
            Data["Courants_Alimentations"]["Référence_GR1"] = (
                Data["Courants_Alimentations"]["Référence_A1"]
                + Data["Courants_Alimentations"]["Référence_A2"]
            )
            Keys.append("Courants_Alimentations/Référence_GR1")
            Groups["Courants_Alimentations"]["Référence_GR1"] = Groups[
                "Courants_Alimentations"
            ]["Référence_A1"]
        if "Référence_A3" in Data["Courants_Alimentations"]:
            Data["Courants_Alimentations"]["Référence_GR2"] = (
                Data["Courants_Alimentations"]["Référence_A3"]
                + Data["Courants_Alimentations"]["Référence_A4"]
            )
            Keys.append("Courants_Alimentations/Référence_GR2")
            Groups["Courants_Alimentations"]["Référence_GR2"] = Groups[
                "Courants_Alimentations"
            ]["Référence_A3"]

        return TdmsMagnetData(name, Groups, Keys, Data, defs_file=defs_file)

    @classmethod
    def fromtxt(cls, name: str, defs_file: str | None = None) -> "PandasMagnetData":
        """Create from a pupitre .txt file."""
        validate_txt_format(name)
        with open(name) as f:
            f_extension = os.path.splitext(name)[-1]
            if f_extension == ".txt":
                Data = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=1)
                Keys = Data.columns.values.tolist()
            else:
                raise RuntimeError(f"fromtxt: expect a txt filename - got {name}")
        return PandasMagnetData(name, {}, Keys, Data, defs_file=defs_file)

    @classmethod
    def fromensight(cls, name: str, defs_file: str | None = None) -> "EnsightMagnetData":
        """Create from a CSV ensight file."""
        validate_file_exists(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=2)
            Keys = Data.columns.values.tolist()
        return EnsightMagnetData(name, {}, Keys, Data, defs_file=defs_file)

    @classmethod
    def fromcsv(cls, name: str, defs_file: str | None = None) -> "PandasMagnetData":
        """Create from a CSV file."""
        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=0)
            Keys = Data.columns.values.tolist()
        return PandasMagnetData(name, {}, Keys, Data, defs_file=defs_file)

    @classmethod
    def fromStringIO(  # noqa: N802
        cls, name: str, sep: str = r"\s+", skiprows: int = 1, defs_file: str | None = None
    ) -> "PandasMagnetData":
        """Create from a StringIO / in-memory string."""
        from io import StringIO

        Data = pd.DataFrame()
        Keys: list[str] = []
        try:
            Data = pd.read_csv(
                StringIO(name), sep=sep, engine="python", skiprows=skiprows
            )
            Keys = Data.columns.values.tolist()
        except (pd.errors.ParserError, ValueError, OSError):
            logger.error("magnetdata.fromStringIO: trouble loading data")
            with open("wrongdata.txt", "w", newline="\n") as fo:
                fo.write(name)

        return PandasMagnetData("stringIO", {}, Keys, Data, defs_file=defs_file)

    @classmethod
    def frombprofile(cls, name: str, defs_file: str | None = None) -> "BProfileMagnetData":
        """Create from a bprofile CSV file (Index, Position, Profile columns)."""
        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=r"\s+", engine="python", skiprows=0)
            Keys = Data.columns.values.tolist()
        return BProfileMagnetData(name, {}, Keys, Data, defs_file=defs_file)

    @classmethod
    def fromfeelpp(cls, name: str, skiprows: int = 0, defs_file: str | None = None) -> "FeelppMagnetData":
        """Create from a feelpp simulation CSV file."""
        validate_csv_format(name)
        with open(name) as f:
            Data = pd.read_csv(f, sep=",", engine="python", skiprows=skiprows)
            Keys = Data.columns.values.tolist()
        return FeelppMagnetData(name, {}, Keys, Data, defs_file=defs_file)
