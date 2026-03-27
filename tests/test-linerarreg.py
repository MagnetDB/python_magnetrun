import numpy as np
import pytest

piecewise_regression = pytest.importorskip("piecewise_regression")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def test_piecewise_regression():
    alpha_1 = -4
    alpha_2 = -2
    constant = 100
    breakpoint_1 = 7
    n_points = 200
    np.random.seed(0)
    xx = np.linspace(0, 20, n_points)
    yy = (
        constant
        + alpha_1 * xx
        + (alpha_2 - alpha_1) * np.maximum(xx - breakpoint_1, 0)
        + np.random.normal(size=n_points)
    )

    pw_fit = piecewise_regression.Fit(xx, yy, start_values=[5], n_breakpoints=1)
    pw_fit.summary()

    pw_fit.plot_data(color="grey", s=20)
    pw_fit.plot_fit(color="red", linewidth=4)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.close()
