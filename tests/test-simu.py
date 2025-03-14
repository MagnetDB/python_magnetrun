"""
idea from chatgpt
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt

from python_magnetrun.MagnetRun import MagnetRun
# from .processing.smoothers import savgol


def addtime(mdata: MagnetData, group: str, channel: str) -> pd.DataFrame:
    import datetime

    print("addtime")

    df = pd.DataFrame(mdata.Data[group][channel])
    t0 = mdata.Groups[group][channel]["wf_start_time"]
    dt = mdata.Groups[group][channel]["wf_increment"]
    df["t"] = [i * dt for i in df.index.to_list()]

    df = df.set_index("t")
    print(df.head())
    return df


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_file",
    help="enter input file (ex: ~/M9_2024.05.13---16_30_51.txt)",
)
args = parser.parse_args()
print(f"args: {args}", flush=True)

file = args.input_file

filename = os.path.basename(file)
site = filename.split("_")[0]
insert = "tutu"

f_extension = os.path.splitext(file)[-1]
print(f"extension: {f_extension}")
if f_extension == ".txt":
    mrun = MagnetRun.fromtxt(site, insert, file)
elif f_extension == ".tdms":
    mrun = MagnetRun.fromtdms(site, insert, file)
else:
    raise RuntimeError(f"{file}: unsupported format")

mdata = mrun.getMData()
print(f"keys: {mdata.getKeys()}")


if f_extension == ".txt":
    Vh = mdata.Data["UH"]
    print(f"Vh: {Vh.info(verbose=True)}, {Vh.head()}, {Vh.describe()}", flush=True)

    Ih = mdata.Data["IH"]

    Vb = mdata.Data["UB"]
    Ib = mdata.Data["IB"]
    timestep = 1  # for txt

if f_extension == ".tdms":
    # Uh = mdata.Data["Tensions_Aimant"]["ALL_internes"]
    if "ALL_internes" in mdata.Data["Tensions_Aimant"]:
        Vh = addtime(mdata, "Tensions_Aimant", "ALL_internes")
        timestep = mdata.Groups["Tensions_Aimant"]["ALL_internes"]["wf_increment"]

    # print(f"Uh({t})=", end="", flush=True)
    if "Courant_GR1" in mdata.Data["Courants_Alimentations"]:
        Ih = addtime(mdata, "Courants_Alimentations", "Courant_GR1")
        timestep = mdata.Groups["Courants_Alimentations"]["Courant_GR1"]["wf_increment"]

    if "ALL_externes" in mdata.Data["Tensions_Aimant"]:
        Vb = addtime(mdata, "Tensions_Aimant", "ALL_externes")
        timestep = mdata.Groups["Tensions_Aimant"]["ALL_externes"]["wf_increment"]

    if "Courant_GR2" in mdata.Data["Courants_Alimentations"]:
        Ib = addtime(mdata, "Courants_Alimentations", "Courant_GR2")
        timestep = mdata.Groups["Courants_Alimentations"]["Courant_GR2"]["wf_increment"]

plt.plot(Vh.index.values, Vh.values, "r", label="Vh")
plt.plot(Vb.index.values, Vb.values, "g", label="Vb")
plt.grid()
plt.legend()
plt.ylabel("V")
plt.xlabel("t[s]")
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
    return np.array([dI1_dt, dI2_dt])


Ih0 = Ih.values[0][0]
Ib0 = Ib.values[0][0]
print(f"Ih0={Ih0}")

# init values from I=31kA from commisionning calculation (see magnet_workflow)
L1, R1, L2, R2, M = [0.002315, 0.01364205, 0.01254, 0.01674274, 0.001965]
det = L1 * L2 - M * M
print(f"Tau1 = {L1/R1} s, Tau2 = {L2/R2} s")
from scipy import linalg

eigenval = linalg.eigvals(
    np.array([[L2 * R1 / det, -M * R1 / det], [-M * R2 / det, L1 * R2 / det]])
)
print(f"eigenval={eigenval}")

simulations, infodict = odeint(
    coupled_rl_circuits,
    np.array([Ih0, Ib0]),
    time_points,
    args=(L1, R1, L2, R2, M),
    full_output=True,
)
print(f"simulations={type(simulations)}, shape={simulations.shape}")
print(infodict)

I1_model = simulations[:, 0]
print(f"I1_model: type={type(I1_model)}, shape={I1_model.shape}")
I2_model = simulations[:, 1]
print(f"I1_model: type={type(I1_model)} shape={I1_model.shape}")
Ih_observ = Ih.to_numpy()
print(f"I1_observ: type={type(Ih_observ)} shape={Ih_observ.shape}")
Ib_observ = Ib.to_numpy()
print(f"I2_observ: type={type(Ib_observ)} shape={Ib_observ.shape}", flush=True)
err_I1 = I1_model - Ih_observ
err_I2 = I2_model - Ib_observ

L2_err_I1 = np.linalg.norm(err_I1) / np.linalg.norm(Ih_observ)
L2_err_I2 = np.linalg.norm(err_I2) / np.linalg.norm(Ib_observ)
print(
    f"err_I1={L2_err_I1}, err_I2={L2_err_I2}",
    flush=True,
)

plt.plot(
    time_points,
    I1_model,
    "or",
    label="num Ih",
    markevery=200,
    linestyle="None",
)
plt.plot(
    time_points,
    I2_model,
    "og",
    label="num Ib",
    markevery=200,
    linestyle="None",
)
plt.plot(Ih.index.values, Ih.values, "r", label="Ih")
plt.plot(Ib.index.values, Ib.values, "g", label="Ib")
plt.grid()
plt.legend()
plt.ylabel("A")
plt.xlabel("t[s]")
plt.show()
