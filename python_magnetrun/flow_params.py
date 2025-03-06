"""
Extract flow params from records using a fit
"""

import tempfile
import os
import re

import numpy as np
from scipy import optimize
from math import floor

import datetime

import json
import pandas as pd
from rich.progress import track
from . import utils

from .utils.fit import fit
from .utils.files import concat_files
from .utils.plots import plot_files
from .magnetdata import MagnetData


def stats(
    Ikey: str,
    Okey: str,
    threshold: float,
    files: list,
    wd: str,
    filename: str,
    debug: bool = False,
):
    df = concat_files(files, keys=[Ikey, Okey], debug=debug)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # # drop values for Icoil1 > Imax
    result = df.query(f"{Ikey} <= {threshold}")  # , inplace=True)
    if result is not None and debug:
        print(f"df: nrows={df.shape[0]}, results: nrows={result.shape[0]}")
        print(f"result max: {result[Ikey].max()}")

    plot_files(
        filename,
        files,
        key1=Ikey,
        key2=Okey,
        fit=None,
        show=debug,
        debug=debug,
        wd=wd,
    )

    if debug:
        stats = result[Okey].describe(include="all")
        print(f"{Okey} stats:\n{stats}")
    return (result[Okey].mean(), result[Okey].std())


def compute(
    files: list,
    Ikey: str,
    RpmKey: str,
    QKey: str,
    PinKey: str,
    PoutKey: str,
    name: str,
    debug: bool = False,
):
    """
    compute flow_params for a given magnet
    """

    cwd = os.getcwd()
    print(f"cwd={cwd}")

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

    Imax = flow_params["Imax"]["value"]  # TODO find Imax value

    def vpump_func(x, a: float, b: float):
        return a * (x / Imax) ** 2 + b

    params = fit(
        Ikey,
        RpmKey,
        "Rpm",
        Imax,
        vpump_func,
        files,
        cwd,
        name,
        debug,
    )
    flow_params["Vp0"]["value"] = params[1]
    flow_params["Vpmax"]["value"] = params[0]
    vp0 = flow_params["Vp0"]["value"]
    vpmax = flow_params["Vpmax"]["value"]
    params = []

    # Fit for Flow
    def flow_func(x, a: float, b: float):
        return a + b * vpump_func(x, vpmax, vp0) / (vpmax + vp0)

    params = fit(
        Ikey,
        QKey,
        "Flow",
        Imax,
        flow_func,
        files,
        cwd,
        name,
        debug,
    )
    flow_params["F0"]["value"] = params[0]
    flow_params["Fmax"]["value"] = params[1]
    params = []

    # Fit for Pressure
    def pressure_func(x, a: float, b: float):
        return a + b * (vpump_func(x, vpmax, vp0) / (vpmax + vp0)) ** 2

    params = fit(
        Ikey,
        PinKey,
        "Pin",
        Imax,
        pressure_func,
        files,
        cwd,
        name,
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
        files,
        cwd,
        name,
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
