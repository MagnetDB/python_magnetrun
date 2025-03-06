import numpy as np
import pandas as pd


from scipy import optimize
from math import floor

from .plots import plot_df
from .files import concat_files


def fit(
    Ikey: str,
    Okey: str,
    Ostring: str,
    threshold: float,
    fit_function,
    df: pd.DataFrame,
    wd: str,
    filename: str,
    show: bool = False,
    debug: bool = False,
):
    """
    perform fit
    """

    result = df.query(f"{Ikey} <= {threshold}")  # , inplace=True)
    if result is not None and debug:
        print(f"df: nrows={df.shape[0]}, results: nrows={result.shape[0]}")
        print(f"result max: {result[Ikey].max()}")

    x_data = result[f"{Ikey}"].to_numpy()
    y_data = result[Okey].to_numpy()
    params, params_covariance = optimize.curve_fit(fit_function, x_data, y_data)

    print(f"{Ostring} Fit:")
    print(f"\tparams: {params}")
    # print(f"\tcovariance: {params_covariance}")
    print(f"\tstderr: {np.sqrt(np.diag(params_covariance))}")

    plot_df(
        filename,
        df,
        key1=Ikey,
        key2=Okey,
        fit=(x_data, [fit_function(x, params[0], params[1]) for x in x_data]),
        show=show,
        debug=debug,
        wd=wd,
    )

    return (params, params_covariance)
