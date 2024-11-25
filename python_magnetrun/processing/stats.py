"""Main module."""

import pandas as pd
import numpy as np

numpy_version = np.__version__.split('.')
if numpy_version[0] == 1:
    numpy_NaN = np.NaN
else:
    numpy_NaN = np.nan
    
print(f'numpy: {numpy_version}', flush=True)

from ..magnetdata import MagnetData
from ..utils.sequence import list_sequence, list_duplicates_of


def stats(Data: MagnetData, fields: list[str] | None = None, debug: bool = False):
    """compute stats from the actual run"""

    # TODO:
    # add teb,... to list
    # add duration
    # add duration per Field above certain values
    # add \int Power over time

    from tabulate import tabulate

    # see https://github.com/astanin/python-tabulate for tablefmt
    if isinstance(Data.Data, pd.DataFrame):
        print("Statistics:\n")
        # print(f"data keys: {Data.getKeys()}", flush=True)
        tables = []
        headers = ["Name", "Mean", "Max", "Min", "Std", "Median", "Mode"]
        selected_fields = ["Field", "Pmagnet", "Ptot", "teb", "tsb", "debitbrut"]
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
                except:
                    # print(f"{f}: failed to compute df.mode()")
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

        print(tabulate(tables, headers, tablefmt="simple"), "\n")
    else:
        for group, df in Data.Data.items():
            print(f"stats for {group}: ", flush=True)
            print(tabulate(df.describe(), headers="keys", tablefmt="psql"), "\n")

    return 0
