"""
idea from chatgpt
"""

import os
import pandas as pd

from python_magnetrun..MagnetRun import MagnetRun
from python_magnetrun..processing.smoothers import savgol

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_file", help="enter input file (ex: ~/M9_Overview_240509-1634.tdms)"
)
parser.add_argument("--window", help="rolling window size", type=int, default=10)
args = parser.parse_args()
print(f"args: {args}", flush=True)

file = args.input_file
window = args.window

filename = os.path.basename(file)
site = filename.split("_")[0]
insert = "tutu"
mrun = MagnetRun.fromtdms(site, insert, file)
mdata = mrun.getMData()
print(f"duration: {mdata.getDuration()}")

# Helices insert

# Uh = mdata.Data["Tensions_Aimant"]["ALL_internes"]
if "ALL_internes" in mdata.Data["Tensions_Aimant"]:
    Vh = mdata.Data["Tensions_Aimant"][["ALL_internes", "t"]]
    Vh = Vh.set_index("t")
else:
    Vh = mdata.Data["Tensions_Aimant"][["ALL internes", "t"]]
    Vh = Vh.set_index("t")
    Vh.rename(columns={"ALL internes": "ALL_internes"}, inplace=True)

print(f"Vh: {Vh.head()}", flush=True)

# print(f"Uh({t})=", end="", flush=True)
if "Courant_GR1" in mdata.Data["Courants_Alimentations"]:
    Ih = mdata.Data["Courants_Alimentations"][["Courant_GR1", "t"]]
    Ih = Ih.set_index("t")
else:
    Ih = mdata.Data["Courants_Alimentations"][["Courant GR1", "t"]]
    Ih = Ih.set_index("t")
    Ih.rename(columns={"Courant GR1": "Courant_GR1"}, inplace=True)

Uh_smoothed = savgol(
    y=Vh["ALL_internes"].to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
print(f"Uh: {Uh_smoothed.shape}")
Ih_smoothed = savgol(
    y=Ih["Courant_GR1"].to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
print(f"Ih: {Ih_smoothed.shape}")
dIh_smoothed = savgol(
    y=Ih["Courant_GR1"].to_numpy(),
    window=window,
    polyorder=3,
    deriv=1,
)
print(f"Ih: {Ih.head()}", flush=True)

# Bitters
# Ub = mdata.Data["Tensions_Aimant"]["ALL_externes"]
if "ALL_externes" in mdata.Data["Tensions_Aimant"]:
    Vb = mdata.Data["Tensions_Aimant"][["ALL_externes", "t"]]
    Vb = Vb.set_index("t", "ALL_externes")
else:
    Vb = mdata.Data["Tensions_Aimant"][["ALL externes", "t"]]
    Vb = Vb.set_index("t", "ALL externes")
    Vb.rename(columns={"ALL externes": "ALL_externes"}, inplace=True)


if "Courant_GR1" in mdata.Data["Courants_Alimentations"]:
    Ib = mdata.Data["Courants_Alimentations"][["Courant_GR2", "t"]]
    Ib = Ib.set_index("t", "Courant_GR2")
else:
    Ib = mdata.Data["Courants_Alimentations"][["Courant GR2", "t"]]
    Ib = Ib.set_index("t", "Courant GR2")
    Ib.rename(columns={"Courant GR2": "Courant_GR2"}, inplace=True)

Ub_smoothed = savgol(
    y=Vb["ALL_externes"].to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
print(f"Ub: {Ub_smoothed.shape}")
Ib_smoothed = savgol(
    y=Ib["Courant_GR2"].to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
print(f"Ib: {Ib_smoothed.shape}")
dIb_smoothed = savgol(
    y=Ib["Courant_GR2"].to_numpy(),
    window=window,
    polyorder=3,
    deriv=1,
)


def estimate_RL(U1, I1, dI1, U2, I2, dI2):
    # Number of data points
    N = len(I1)

    # Construct A matrix and b vector
    A = np.zeros((2 * (N - 1), 5))
    b = np.zeros(2 * (N - 1))

    for i in range(N - 1):
        A[2 * i] = [I1[i], dI1[i], 0, 0, dI2[i]]
        A[2 * i + 1] = [0, 0, I2[i], dI2[i], dI1[i]]

        b[2 * i] = U1[i]
        b[2 * i + 1] = U2[i]

    # Solve the linear system
    res = lstsq(A, b)
    print(f"estimate_L: {res}")

    # Extract parameters
    R1, L1, R2, L2, M = res[0]
    return R1, L1, R2, L2, M


R1, L1, R2, L2, M = estimate_RL(
    Uh_smoothed,
    Ih_smoothed,
    dIh_smoothed * 120,
    Ub_smoothed,
    Ib_smoothed,
    dIb_smoothed * 120,
)
print(f"Rh={R1}, Lh={L1}, Rb={R2}, Lb={L2}, M={M}")

data = {
    "Uh": Uh_smoothed,
    "Ub": Ub_smoothed,
    "est_Uh": R1 * Ih_smoothed + L1 * dIh_smoothed * 120 + M * dIb_smoothed * 120,
    "est_Ub": R2 * Ib_smoothed + L2 * dIb_smoothed * 120 + M * dIh_smoothed * 120,
}
Y = pd.DataFrame(data)
ax = Y.plot()

plt.grid()
plt.legend()
plt.show()


from scipy.integrate import odeint
from scipy.optimize import least_squares

time_points = Vh.index.values
print(f"time_points: {time_points.shape}")


# Interpolation function
from scipy.interpolate import interp1d

Vh_interpolator = interp1d(
    time_points,
    Vh["ALL_internes"].values,
    kind="linear",
    fill_value="extrapolate",
)
Vb_interpolator = interp1d(
    time_points,
    Vb["ALL_externes"].values,
    kind="linear",
    fill_value="extrapolate",
)

Ih_interpolator = interp1d(
    time_points,
    Ih["Courant_GR1"].values,
    kind="linear",
    fill_value="extrapolate",
)
Ib_interpolator = interp1d(
    time_points,
    Ib["Courant_GR2"].values,
    kind="linear",
    fill_value="extrapolate",
)


# Example voltage functions
def V1(t):
    # print(f"Vh({t})=", end="", flush=True)
    value = Vh_interpolator(t)  # , .at[t, "ALL_internes"]
    # print(value)
    return value


def V2(t):
    # print(f"Vb({t})=", end="", flush=True)
    value = Vb_interpolator(t)  # Vb.iat[idx]  # , .at[t, "ALL_externes"]
    # print(value)
    return value


def I1(t):
    # print(f"Vh({t})=", end="", flush=True)
    value = Ih_interpolator(t)  # , .at[t, "ALL_internes"]
    # print(value)
    return value


def I2(t):
    # print(f"Vb({t})=", end="", flush=True)
    value = Ib_interpolator(t)  # Vb.iat[idx]  # , .at[t, "ALL_externes"]
    # print(value)
    return value


# Define the coupled ODEs
def coupled_rl_circuits(y, t, L1, R1, L2, R2, M):
    # print(f"coupled_rl_circuits: t={t}, y={y}, V1={V1(t)}, V2={V2(t)}", flush=True)
    I1, I2 = y

    # dIdt = 1/L ( V - R*I)
    # with:
    # 1/L = 1/(L1*L2-M*M) * [[L2, -M], [-M, L1]]
    det_L = L1 * L2 - M * M
    dI1_dt = 1 / det_L * (L2 * (V1(t) - R1 * I1) - M * (V2(t) - R2 * I2))
    dI2_dt = 1 / det_L * (-M * (V1(t) - R1 * I1) + L1 * (V2(t) - R2 * I2))
    return [dI1_dt, dI2_dt]


Ih0_smoothed = Ih_smoothed[0]
Ib0_smoothed = Ib_smoothed[0]


def residuals(parameters, time, actual_position):
    # print(f"residuals: time={type(time)}, time={time}", flush=True)
    L1, R1, L2, R2, M = parameters
    simulated_position, infodict = odeint(
        coupled_rl_circuits,
        [Ih0_smoothed, Ib0_smoothed],
        time,
        args=(L1, R1, L2, R2, M),
        full_output=True,
    )
    print(
        f"residuals: simulated_position={type(simulated_position)}, shape={simulated_position.shape}"
    )
    print(f"residuals: infodict={infodict}")

    [I1_model, I2_model] = np.transpose(simulated_position)
    err_I1 = np.linalg.norm(I1_model - actual_position[0]) / np.linalg.norm(
        actual_position[0]
    )
    err_I2 = np.linalg.norm(I2_model - actual_position[1]) / np.linalg.norm(
        actual_position[1]
    )
    print(
        f"residuals: err_I1={err_I1}, err_I2={err_I2}",
        flush=True,
    )

    res = np.transpose(simulated_position) - actual_position
    print(f"residuals: res={type(res)}, shape={res.shape}")
    return np.transpose(simulated_position) - actual_position


# Initial guesses for L1, R1, L2, R2, M
initial_guess = [0.002315, R1, 0.01254, R2, 0.001965]

optimization_result = least_squares(
    residuals,
    initial_guess,
    args=(
        time_points,
        [Ih["Courant_GR1"].values, Ib["Courant_GR2"].values],
    ),
)
# Extract the estimated parameters
L1_est, R1_est, L2_est, R2_est, M_est = optimization_result.x


# Extract estimated parameters
print("Estimated Parameters:\n")
print(f"Rh={R1_est}, Lh={L1_est}, Rb={R2_est}, Lb={L2_est}, M={M_est}")


Y["scipy_Uh"] = R1_est * Ih_smoothed + L1_est * dIh_smoothed + M_est * dIb_smoothed
Y["scipy_Ub"] = R2_est * Ib_smoothed + L2_est * dIb_smoothed + M_est * dIh_smoothed

ax = Vh.plot()
Vb.plot(ax=ax)
plt.plot(Vb.index, Y["Uh"], "r", label="smoothed_Uh")
plt.plot(Vb.index, Y["Ub"], "g", label="smoothed_Ub")
plt.plot(
    Vb.index, Y["scipy_Uh"], "or", label="scipy_Uh", markevery=100, linestyle="None"
)
plt.plot(
    Vb.index, Y["scipy_Ub"], "og", label="scipy_Ub", markevery=100, linestyle="None"
)

plt.grid()
plt.legend()
plt.show()
