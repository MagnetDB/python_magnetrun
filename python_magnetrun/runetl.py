"""ETL functions for preparing MagnetRun data."""

import logging
import re
import warnings

from natsort import natsorted

from .magnetdata import MagnetData

logger = logging.getLogger(__name__)


def prepareData_legacy(data: MagnetData, housing: str) -> None:
    """Prepare magnet run data by adding computed fields and renaming columns
    (LEGACY VERSION).

    Adds IH_ref/IB_ref computed currents and renames Flow/Rpm/Tin/HP columns
    with H/B suffixes appropriate for the given housing configuration.

    :param data: MagnetData object to prepare in-place
    :type data: MagnetData
    :param housing: Housing name (e.g. "M8", "M9", "M10")
    :type housing: str
    """
    warnings.warn(
        "prepareData_legacy is deprecated and will be removed in a future version. "
        "Use prepareData instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # get start/end
    (start_date, start_time, end_date, end_time) = data.getStartDate()
    logger.debug(
        f"prepareData_legacy: start_date={start_date}, start_time={start_time}, end_date={end_date}, end_time={end_time}"  # noqa: E501
    )

    # add timestamp
    data.addTime()

    # get duration
    _duration = data.getDuration()

    # TODO use a dict struct to simplify this?
    # shall check if key exist beforehand
    if housing == "M9":
        data.addData("IH_ref", "IH_ref = Idcct1 + Idcct2")
        data.addData("IB_ref", "IB_ref = Idcct3 + Idcct4")

        # FlowH = Flow1, FlowB = Flow2
        for field in ["Flow", "Rpm", "Tin", "HP"]:
            data.renameData(
                columns={f"{field}1": f"{field}H", f"{field}2": f"{field}B"}
            )

    elif housing in ["M8", "M10"]:
        data.addData("IH_ref", "IH_ref = Idcct3 + Idcct4")
        data.addData("IB_ref", "IB_ref = Idcct1 + Idcct2")

        # FlowH = Flow2, FlowB = Flow1
        for field in ["Flow", "Rpm", "Tin", "HP"]:
            data.renameData(
                columns={f"{field}1": f"{field}B", f"{field}2": f"{field}H"}
            )
    # what about M1, M5 and M7???

    data.cleanupData_legacy()
    Ikey = natsorted([_key for _key in data.getKeys() if re.match(r"Icoil\d+", _key)])
    logger.debug(f"MagnetRun/prepareData_legacy: housing={housing}, Ikey={Ikey}")

    data.renameData(columns={f"{Ikey[0]}": "IH"})
    data.renameData(columns={f"{Ikey[-1]}": "IB"})

    logger.debug(f"MagnetRun.prepareData_legacy: data.keys={data.getKeys()}")


def prepareData(
    data: MagnetData,
    housing: str,
    keys_to_remove: list[str] | None = None,
    keys_to_rename: dict[str, str] | None = None,
    keys_to_add: dict[str, str] | None = None,
    debug: bool = False,
) -> None:
    """Prepare magnet run data by adding computed fields and renaming columns.

    This method adds timestamp and performs cleanup with flexible configuration.
    Housing-specific operations (IH_ref/IB_ref, Flow/Rpm/Tin/HP renaming, Icoil→IH/IB)
    should now be specified via the keys_to_add and keys_to_rename parameters.

    All custom operations are handled by the cleanupData() method.

    :param data: MagnetData object to prepare in-place
    :type data: MagnetData
    :param housing: Housing name (e.g. "M8", "M9", "M10") - for reference/logging
    :type housing: str
    :param keys_to_remove: list of column names to remove, defaults to None
    :type keys_to_remove: list[str] | None, optional
    :param keys_to_rename: dict mapping old column names to new names, defaults to None
    :type keys_to_rename: dict[str, str] | None, optional
    :param keys_to_add: dict mapping new column names to their formulas,
        defaults to None
    :type keys_to_add: dict[str, str] | None, optional
    :param debug: Enable debug output, defaults to False
    :type debug: bool, optional
    """
    # get start/end
    (start_date, start_time, end_date, end_time) = data.getStartDate()
    logger.debug(
        f"prepareData: start_date={start_date}, start_time={start_time}, end_date={end_date}, end_time={end_time}"  # noqa: E501
    )

    # add timestamp
    data.addTime()

    # get duration
    _duration = data.getDuration()

    # NOTE: All custom operations (keys_to_add, keys_to_rename, keys_to_remove)
    # are now handled by cleanupData() method

    # Perform cleanup with flexible parameters
    data.cleanupData(
        keys_to_remove=keys_to_remove,
        keys_to_rename=keys_to_rename,
        keys_to_add=keys_to_add,
        debug=debug,
    )

    logger.debug(f"MagnetRun.prepareData: data.keys={data.getKeys()}")
