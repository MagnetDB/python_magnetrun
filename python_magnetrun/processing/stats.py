"""Main module."""

import pandas as pd
import numpy as np

numpy_version = np.__version__.split(".")
if numpy_version[0] == 1:
    numpy_NaN = np.NaN
else:
    numpy_NaN = np.nan


from ..magnetdata import MagnetData
from ..utils.sequence import list_sequence, list_duplicates_of

from tabulate import tabulate


def stats(
    Data: MagnetData,
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
    if isinstance(Data.Data, pd.DataFrame):
        # print(f"data keys: {Data.getKeys()}", flush=True)
        tables = []
        headers = ["Name", "Mean", "Max", "Min", "Std", "Median", "Mode"]
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
                df = Data.Data[f]
                if debug:
                    print(f"get stats for {f} ({Data.getKeys()})", flush=True)
                    print(f"{f}: {df.head()}", flush=True)
                v_min = float(df.min())
                v_max = float(df.max())
                v_mean = float(df.mean())
                v_std = float(df.std())
                v_median = float(df.median())
                v_mode = numpy_NaN  # Most frequent value in a data set
                try:
                    v_mode = float(df.mode().iloc[0])
                except Exception as e:
                    if debug:
                        print(f"{f}: failed to compute df.mode() - {e}")
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
        for group, df in Data.Data.items():
            print(f"stats for {group}: ", flush=True)
            tables = df.describe()
            headers = "keys"

    if display:
        print("Statistics:\n")
        print(tabulate(tables, headers, tablefmt=fmt), "\n")

    return (tables, headers)
