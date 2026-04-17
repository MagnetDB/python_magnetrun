"""Main module."""

import logging

import numpy as np
import pandas as pd
from tabulate import tabulate  # type: ignore[import-untyped]

from ..magnetdata_base import DataType, MagnetDataBase

logger = logging.getLogger(__name__)

numpy_version = np.__version__.split(".")
numpy_NaN = np.NaN if numpy_version[0] == 1 else np.nan


def stats(
    Data: MagnetDataBase,
    fields: list[str] | None = None,
    fmt: str = "simple",
    display: bool = True,
    debug: bool = False,
) -> tuple:
    """compute stats from the actual run"""

    # TODO:
    # add teb,... to list
    # add duration
    # add duration per Field above certain values
    # add \int Power over time
    # fmt: "plain", "simple", "psql"

    # see https://github.com/astanin/python-tabulate for tablefmt
    if Data.Type != DataType.TDMS:
        # print(f"data keys: {Data.getKeys()}", flush=True)
        tables: list | pd.DataFrame = []
        headers: list[str] | str = [
            "Name",
            "Mean",
            "Max",
            "Min",
            "Std",
            "Median",
            "Mode",
        ]
        selected_fields = [
            "Field",
            "IH",
            "IB",
            "Pmagnet",
            "Ptot",
            "TAlimout",
            "teb",
            "tsb",
            "debitbrut",
        ]
        if fields is not None:
            selected_fields = fields

        for f in selected_fields:
            table = [
                f"{f}[N/A]",
                numpy_NaN,
                numpy_NaN,
                numpy_NaN,
                numpy_NaN,
                numpy_NaN,
                numpy_NaN,
            ]
            if f in Data.getKeys():
                fname, unit = Data.getUnitKey(f)
                df = Data.getData([f])[f]
                logger.debug(f"get stats for {f} ({Data.getKeys()})")
                logger.debug(f"{f}: {df.head()}")
                v_min = float(df.min())
                v_max = float(df.max())
                v_mean = float(df.mean())
                v_std = float(df.std())
                v_median = float(df.median())
                v_mode = numpy_NaN  # Most frequent value in a data set
                try:
                    v_mode = float(df.mode().iloc[0])
                except (IndexError, ValueError) as e:
                    logger.debug(f"{f}: failed to compute df.mode() - {e}")
                    pass
                table = [
                    f"{f}[{unit:~P}]",
                    v_mean,
                    v_max,
                    v_min,
                    v_std,
                    v_median,
                    v_mode,
                ]

            tables.append(table)

    else:
        for group in Data.Groups:
            df = Data.getData(group)
            logger.info(f"stats for {group}: ")
            tables = df.describe()
            headers = "keys"

    if display:
        logger.info("Statistics:")
        logger.info(f"{tabulate(tables, headers, tablefmt=fmt)}")

    return (tables, headers)
