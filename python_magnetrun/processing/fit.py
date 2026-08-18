import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from ..plotting.backend import get_backend
from ..plotting.timeseries import plot_xy

logger = logging.getLogger(__name__)


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
        logger.debug(f"df: nrows={df.shape[0]}, results: nrows={result.shape[0]}")
        logger.debug(f"result max: {result[Ikey].max()}")

    x_data = result[f"{Ikey}"].to_numpy()
    y_data = result[Okey].to_numpy()
    params, params_covariance = optimize.curve_fit(fit_function, x_data, y_data)

    logger.info(f"Fit: {Ostring}({Ikey}) ")
    logger.info(f"\tparams: {params}")
    logger.info(f"\tstderr: {np.sqrt(np.diag(params_covariance))}")

    b = get_backend("matplotlib")
    fig = plot_xy(
        df, [(Ikey, Okey)],
        backend=b,
        title=f"{filename}: {Okey}({Ikey})",
        marker="o",
        linestyle="none",
    )
    fit_y = np.array([fit_function(x, params[0], params[1]) for x in x_data])
    b.add_series(fig, 0, x_data, fit_y, label="fit", color="red", linestyle="--")
    b.finalize(fig, xlabel=Ikey)
    if show:
        b.show(fig)
    else:
        b.save(fig, Path(wd) / f"{filename}-{Ikey}_vs_{Okey}.png")

    return (params, params_covariance)


def find_eqn(my_pwlf_2):
    from sympy import Symbol
    from sympy.utilities import lambdify

    x = Symbol("x")

    def get_symbolic_eqn(pwlf_, segment_number):
        if pwlf_.degree < 1:
            raise ValueError("Degree must be at least 1")
        if segment_number < 1 or segment_number > pwlf_.n_segments:
            raise ValueError("segment_number not possible")
        # assemble degree = 1 first
        for line in range(segment_number):
            if line == 0:
                my_eqn = pwlf_.beta[0] + (pwlf_.beta[1]) * (x - pwlf_.fit_breaks[0])
            else:
                my_eqn += (pwlf_.beta[line + 1]) * (x - pwlf_.fit_breaks[line])
        # assemble all other degrees
        if pwlf_.degree > 1:
            for k in range(2, pwlf_.degree + 1):
                for line in range(segment_number):
                    beta_index = pwlf_.n_segments * (k - 1) + line + 1
                    my_eqn += (pwlf_.beta[beta_index]) * (x - pwlf_.fit_breaks[line]) ** k

        return my_eqn.simplify()

    eqn_list = []
    f_list = []
    coeff_list = []
    for i in range(my_pwlf_2.n_segments):
        eqn_list.append(get_symbolic_eqn(my_pwlf_2, i + 1))
        f_list.append(lambdify(x, eqn_list[-1]))
        coeff_list.append(eqn_list[-1].as_poly().all_coeffs())

    return (eqn_list, coeff_list)
