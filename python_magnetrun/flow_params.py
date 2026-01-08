"""
Extract flow params from records using a fit
"""

import logging
import os

import json
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from tabulate import tabulate
from sympy import Symbol

from .processing.fit import fit, find_eqn
from .utils.plots import plot_df
from typing import List


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
    return (result[Okey].mean().item(), result[Okey].std().item())


def pwlf_fit(
    Ikey,
    x,
    PKey,
    y,
    degree: int = 1,
    segment: int = 1,
    guess: List[float] = None,
    show: bool = False,
    debug: bool = False,
):
    import pwlf

    print(f"pwl_fit: Ikey={Ikey}, PKey={PKey}, show={show}")
    my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=degree)
    if guess is not None and guess:
        res = my_pwlf.fit_guess(guess)
        print(f"pwlf: using guess breaks (guess={guess}): {res}")
    else:
        res = my_pwlf.fit(segment)
    errors = my_pwlf.standard_errors()
    print(f"pwlf: res={res}, errors={errors}")
    xHat = np.linspace(min(x), max(x), num=10000)
    yHat = my_pwlf.predict(xHat)

    # TODO test fit with breakpoint guess for Imax: breaks = my_pwlf.fit_guess([Imax])

    # get error
    p = my_pwlf.p_values(method="non-linear", step_size=1e-4)
    se = my_pwlf.se  # standard errors
    print("pwlf beta: ", my_pwlf.beta)
    parameters = np.concatenate((my_pwlf.beta, my_pwlf.fit_breaks[1:-1]))

    tables = []
    headers = [
        "Parameter type",
        "Parameter value",
        "Standard error",
        "t",
        "P > np.abs(t) (p-value)",
    ]

    values = np.zeros((parameters.size, 5), dtype=np.object_)
    values[:, 1] = np.around(parameters, decimals=3)
    values[:, 2] = np.around(se, decimals=3)
    values[:, 3] = np.around(parameters / se, decimals=3)
    values[:, 4] = np.around(p, decimals=3)

    for i, row in enumerate(values):
        table = []
        if i < my_pwlf.beta.size:
            table.append("Beta")
        else:
            table.append("Breakpoint")
        table += row.tolist()[1:]
        tables.append(table)
    print(tabulate(tables, headers=headers, tablefmt="psql"), flush=True)

    eqn_list = []
    if degree >= 1:
        (eqn_list, coeff_list) = find_eqn(my_pwlf)

    # plot the results

    if show:
        plt.figure()
        plt.plot(x, y, "o")
        plt.plot(xHat, yHat, "-")

        if debug:
            for eqn in eqn_list:
                eqnHat = [eqn.evalf(subs={Symbol("x"): val}) for val in xHat.tolist()]
                plt.plot(xHat, eqnHat, ".", alpha=0.2)
            # set xrange, yrange
            plt.xlim([x.min(), 1.1 * x.max()])
            plt.ylim([y.min(), 1.1 * y.max()])
        plt.title(f"{PKey}({Ikey}): pwlf,  res={res}")
        plt.grid()
        plt.show()
        plt.close()

    return (my_pwlf, eqn_list)


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
    print(
        f"flow_params.compute: Ikey={Ikey}, RpmKey={RpmKey},  Qkey={QKey}, PinKey={PinKey}, PoutKey={PoutKey}"
    )
    cwd = os.getcwd()
    # print(f"cwd={cwd}")
    # print(df.head())

    _df = df.query(f"{Ikey} >= 300")
    # print(_df.head())

    flow_params = setup()
    Imax = flow_params["Imax"]["value"]  # TODO find Imax value

    def vpump_func(x, a: float, b: float):
        return a * (x / Imax) ** 2 + b

    x = _df[Ikey].to_numpy()
    y = _df[RpmKey].to_numpy()
    for segment in [1, 2]:
        my_pwlf, eqns = pwlf_fit(
            Ikey, x, RpmKey, y, degree=2, segment=segment, show=True
        )
        # TODO if error ?my_pwlf.standard_errors()? on brkpoints is big, try with 1 segment
        final_y = eqns[0].evalf(subs={Symbol("x"): x[-1]})
        if abs(final_y - y[-1]) <= 10:
            break

    # compute Imax, Vp0, Vpmax
    print(f"{RpmKey}({Ikey}): {my_pwlf.n_segments}")
    if my_pwlf.n_segments == 2:
        print(f"new_Imax: {Imax} ->{my_pwlf.fit_breaks[1]}")
        Imax = my_pwlf.fit_breaks[1]

    print(
        f'new Vp0={flow_params["Vp0"]["value"]} -> {eqns[0].evalf(subs={Symbol("x"): 0})}'
    )
    print(
        f'new Vpmax={flow_params["Vpmax"]["value"]} -> {eqns[0].evalf(subs={Symbol("x"): Imax})}'
    )
    params = [
        float(eqns[0].evalf(subs={Symbol("x"): Imax})),
        float(eqns[0].evalf(subs={Symbol("x"): 0})),
    ]

    flow_params["Imax"]["value"] = Imax
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

    print(f"{QKey}: params={params}")
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

    print(f"{PinKey}: params={params}")
    flow_params["Pmin"]["value"] = params[0]
    flow_params["Pmax"]["value"] = params[1]
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
    print(f"final flow_params: {flow_params}", flush=True)

    # # save flow_params
    filename = f"{cwd}/{name}-{Ikey}-flow_params.json"
    print(f"{filename}: {json.dumps(flow_params, indent=4)}")
    with open(filename, "w") as f:
        f.write(json.dumps(flow_params, indent=4))


def debitbrut(df: pd.DataFrame, ofile: str, nlevels: int = 4):
    print(f"debitbrut: ofile={ofile}, nlevels={nlevels}")

    # DebitBrut
    qt0 = df.index.values[0]

    qsymbol = "Q"
    psymbol = "P"
    punit = "P"
    from pint import UnitRegistry

    ureg = UnitRegistry()
    qunit = ureg.meter**3 / ureg.hour
    punit = ureg.megawatt

    # use pwlf to get threshold and value
    # TODO how to estimate the number of segment
    # is it enough to get Pmagnet.max to have an idea of segments???
    # or try more advanded features: see find the best number of line segments
    # see https://jekel.me/piecewise_linear_fit_py/examples.html#fit-constants-or-polynomials
    print(f'Pmagnet max: {df["Pmagnet"].max()}')
    # Pmagnet > 15 MW: 7
    # Pmagnet > 10 MW: 5
    # Pmagnet > MW: 3
    # sinon 1
    x = df["t"].to_numpy()
    y = df["debitbrut"].to_numpy()
    df.plot(x="Pmagnet", y="debitbrut")
    plt.title(f"{ofile}: Debitbrut(Pmagnet)")
    plt.grid()
    plt.show()
    plt.close()

    # (changes, regimes, times, values, trend_component) = trends_df(df_pupitre, "t", "debitbrut", args.window, threshold_dict["debitbrut"], overview_dict[ofile]["sources"]["pupitre"], show=True)

    from .processing.hysteresis import (
        multi_level_hysteresis,
        remove_low_x_outliers,
        estimate_hysteresis_parameters,
    )

    # Automatically detect bottom 25% of x, remove outliers there
    df_clean = remove_low_x_outliers(
        df,
        x_col="Pmagnet",
        y_col="debitbrut",
        x_percentile=25,
        verbose=True,
    )

    # Calculate differences between consecutive values
    xdf = df_clean[["t", "debitbrut", "Pmagnet"]].copy()

    # how to estimate levels ? from max pmagnet? from debitbrut?
    print("estimate_hysteresis_parameters:")
    result = estimate_hysteresis_parameters(
        xdf, x_col="Pmagnet", y_col="debitbrut", n_levels=nlevels, verbose=True
    )

    # Extract the parameters for multi_level_hysteresis()
    thresholds = result["thresholds"]
    low_values = result["low_values"]
    high_values = result["high_values"]
    print(f"Estimated thresholds: {thresholds}")
    print(f"Estimated low values: {low_values}")
    print(f"Estimated high values: {high_values}", flush=True)

    x = xdf["Pmagnet"].to_numpy()
    y = xdf["debitbrut"].to_numpy()
    y_model = multi_level_hysteresis(x, thresholds, low_values, high_values)

    # compute error
    residuals = y - y_model
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y - y_model) / y))
    print(f"debitbrut: mae={mae}, mse = {mse}, rmse={rmse}, mape={mape}", flush=True)

    # overview_dict[ofile]["sources"]["pupitre"]
    symbol = "Q_brut"
    yunit = ureg.meter**3 / ureg.hour  # see magnetdata.py L394

    my_ax = plt.gca()
    legends = ["debitbrut"]
    xdf.plot(x="t", y="debitbrut", ax=my_ax)
    legends.append("ymodel")
    my_ax.plot(xdf["t"].to_numpy(), y_model, marker="*", alpha=0.2)
    plt.legend(legends)
    plt.grid()
    plt.title(f"Debitbrut(Pmagnet) model")
    plt.xlabel("t[s]")
    plt.ylabel(f"{symbol} [{yunit:~P}]")
    plt.show()
    plt.close()

    return (thresholds, high_values, low_values)
