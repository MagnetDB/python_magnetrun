"""
Extract flow params from records using a fit
"""

import os

import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from tabulate import tabulate
from sympy import Symbol

from .utils.fit import fit, find_eqn
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


def pwlf_fit(
    Ikey,
    x,
    PKey,
    y,
    degree: int = 1,
    segment: int = 1,
    show: bool = False,
    debug: bool = False,
):
    import pwlf

    my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=degree)
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
        plt.title(f"{Ikey} vs {PKey}: pwlf,  res={res}")
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
    print(f"cwd={cwd}")
    print(df.head())

    _df = df.query(f"{Ikey} >= 300")
    print(_df.head())

    flow_params = setup()
    Imax = flow_params["Imax"]["value"]  # TODO find Imax value

    def vpump_func(x, a: float, b: float):
        return a * (x / Imax) ** 2 + b

    x = _df[Ikey].to_numpy()
    y = _df[RpmKey].to_numpy()
    my_pwlf, eqns = pwlf_fit(Ikey, x, RpmKey, y, 2, 2, show=True)

    # compute Imax, Vp0, Vpmax
    print(f"{Ikey} vs {RpmKey}: {my_pwlf.n_segments}")
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

    x = _df[Ikey].to_numpy()
    y = _df[QKey].to_numpy()
    my_pwlf, eqns = pwlf_fit(Ikey, x, QKey, y, 2, 2, show=True)

    # compute Imax, Vp0, Vpmax
    print(f"{Ikey} vs {QKey}: {my_pwlf.n_segments}")
    if my_pwlf.n_segments == 2:
        print(f"new_Imax: {Imax} ->{my_pwlf.fit_breaks[1]}")
        # Imax = my_pwlf.fit_breaks[1]
        print(
            f'new F0={flow_params["F0"]["value"]} -> {eqns[0].evalf(subs={Symbol("x"): 0})}'
        )
        print(
            f'new Fmax={flow_params["Fmax"]["value"]} -> {eqns[0].evalf(subs={Symbol("x"): Imax})}'
        )
        params = [
            float(eqns[0].evalf(subs={Symbol("x"): Imax})),
            float(eqns[0].evalf(subs={Symbol("x"): 0})),
        ]
        print(params[0], type(params[0]))

    flow_params["Imax"]["value"] = Imax
    flow_params["Fmax"]["value"] = params[0]
    flow_params["F0"]["value"] = params[1]
    params = []

    # Fit for Pressure
    def pressure_func(x, a: float, b: float):
        return a + b * (vpump_func(x, vpmax, vp0) / (vpmax + vp0)) ** 2

    x = _df[Ikey].to_numpy()
    y = _df[PinKey].to_numpy()

    # TODO not so good in general - try with breaks = my_pwlf.fit_guess([Imax])

    my_pwlf, eqns = pwlf_fit(Ikey, x, PinKey, y, 2, 2, show=True)

    # compute Imax, Vp0, Vpmax
    print(f"{Ikey} vs {PinKey}: {my_pwlf.n_segments}")
    if my_pwlf.n_segments == 2:
        print(f"new_Imax: {Imax} ->{my_pwlf.fit_breaks[1]}")
        # Imax = my_pwlf.fit_breaks[1]
        print(
            f'new Pmin={flow_params["Pmin"]["value"]} -> {eqns[0].evalf(subs={Symbol("x"): 0})}'
        )
        print(
            f'new Pmax={flow_params["Pmax"]["value"]} -> {eqns[0].evalf(subs={Symbol("x"): Imax})}'
        )
        params = [
            float(eqns[0].evalf(subs={Symbol("x"): Imax})),
            float(eqns[0].evalf(subs={Symbol("x"): 0})),
        ]
        print(params[0], type(params[0]))

    flow_params["Pmax"]["value"] = params[0]
    flow_params["Pmin"]["value"] = params[1]
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
    filename = f"{cwd}/{name}-{Ikey}-flow_params.json"
    print(f"{filename}: {json.dumps(flow_params, indent=4)}")
    with open(filename, "w") as f:
        f.write(json.dumps(flow_params, indent=4))
