"""
Extract flow params from records using a fit
"""

import os

import json
import pandas as pd
import numpy as np


from .utils.fit import fit
from .utils.plots import plot_df


def setup():
    # default value
    # set Imax to 40 kA to enable real Imax detection
    flow_params = {
        "Vp0": {"value": 1000, "unit": "rpm"},
        "Vpmax": {"value": 2840, "unit": "rpm"},
        "F0": {"value": 0, "unit": "l/s"},
        "Fmax": {"value": 61.71612272405876, "unit": "l/s"},
        "Pmax": {"value": 22, "unit": "bar"},
        "Pmin": {"value": 4, "unit": "bar"},
        "Pout": {"value": 4, "unit": "bar"},
        "Imax": {"value": 28000, "unit": "A"},
    }

    return flow_params


def stats(
    Ikey: str,
    Okey: str,
    threshold: float,
    df: pd.DataFrame,
    wd: str,
    filename: str,
    show: bool = False,
    debug: bool = False,
):
    # # drop values for Icoil1 > Imax
    result = df.query(f"{Ikey} <= {threshold}")  # , inplace=True)
    if result is not None and debug:
        print(f"df: nrows={df.shape[0]}, results: nrows={result.shape[0]}")
        print(f"result max: {result[Ikey].max()}")

    plot_df(
        filename,
        df,
        key1=Ikey,
        key2=Okey,
        fit=None,
        show=show,
        debug=debug,
        wd=wd,
    )

    if debug:
        stats = result[Okey].describe(include="all")
        print(f"{Okey} stats:\n{stats}")
    return (result[Okey].mean(), result[Okey].std())


def compute(
    df: pd.DataFrame,
    Ikey: str,
    RpmKey: str,
    QKey: str,
    PinKey: str,
    PoutKey: str,
    name: str,
    show: bool = False,
    debug: bool = False,
):
    """
    compute flow_params for a given magnet
    """

    cwd = os.getcwd()
    print(f"cwd={cwd}")

    _df = df.query(f"{Ikey} >= 300")

    flow_params = setup()
    Imax = flow_params["Imax"]["value"]  # TODO find Imax value

    def vpump_func(x, a: float, b: float):
        return a * (x / Imax) ** 2 + b

    params, params_covariance = fit(
        Ikey,
        RpmKey,
        "Rpm",
        Imax,
        vpump_func,
        _df,
        cwd,
        name,
        show,
        debug,
    )
    if np.sqrt(np.diag(params_covariance))[0] > 0.1:
        import pwlf

        x = _df[Ikey].to_numpy()
        y = _df[RpmKey].to_numpy()
        my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=2)
        res = my_pwlf.fit(2)
        print(f"pwlf: res={res}")
        xHat = np.linspace(min(x), max(x), num=10000)
        yHat = my_pwlf.predict(xHat)

        # plot the results
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x, y, "o")
        plt.plot(xHat, yHat, "-")
        plt.grid()
        plt.show()
        plt.close()
        # exit(1)

    flow_params["Vp0"]["value"] = params[1]
    flow_params["Vpmax"]["value"] = params[0]
    vp0 = flow_params["Vp0"]["value"]
    vpmax = flow_params["Vpmax"]["value"]
    params = []

    # Fit for Flow
    def flow_func(x, a: float, b: float):
        return a + b * vpump_func(x, vpmax, vp0) / (vpmax + vp0)

    params, params_covariance = fit(
        Ikey,
        QKey,
        "Flow",
        Imax,
        flow_func,
        _df,
        cwd,
        name,
        show,
        debug,
    )
    flow_params["F0"]["value"] = params[0]
    flow_params["Fmax"]["value"] = params[1]
    params = []

    # Fit for Pressure
    def pressure_func(x, a: float, b: float):
        return a + b * (vpump_func(x, vpmax, vp0) / (vpmax + vp0)) ** 2

    params, params_covariance = fit(
        Ikey,
        PinKey,
        "Pin",
        Imax,
        pressure_func,
        _df,
        cwd,
        name,
        show,
        debug,
    )
    flow_params["Pmin"]["value"] = params[0]
    flow_params["Pmax"]["value"] = params[1]
    P0 = flow_params["Pmin"]["value"]
    Pmax = flow_params["Pmax"]["value"]
    params = []

    # correlation Pout
    params = stats(
        Ikey,
        PoutKey,
        Imax,
        _df,
        cwd,
        name,
        show,
        debug,
    )
    print(f"Pout(mean, std): {params}")
    Pout = params[0]
    flow_params["Pout"]["value"] = Pout

    # save flow_params
    print(f"flow_params: {json.dumps(flow_params, indent=4)}")
    filename = f"{cwd}/{name}-flow_params.json"
    with open(filename, "w") as f:
        f.write(json.dumps(flow_params, indent=4))
