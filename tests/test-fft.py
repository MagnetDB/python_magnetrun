"""
idea from chatgpt
"""

import os
import pandas as pd
import statsmodels.api as sm

from math import pi
import numpy as np

from scipy.fft import fft, fftshift, fftfreq
import matplotlib.pyplot as plt

from python_magnetrun.MagnetRun import MagnetRun
# from .processing.smoothers import savgol

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
    Vh = (
        mdata.Data["Ucoil1"]
        + mdata.Data["Ucoil2"]
        + mdata.Data["Ucoil3"]
        + mdata.Data["Ucoil4"]
        + mdata.Data["Ucoil5"]
        + mdata.Data["Ucoil6"]
        + mdata.Data["Ucoil7"]
    )
    print(f"Vh: {Vh.info(verbose=True)}, {Vh.head()}, {Vh.describe()}", flush=True)

    Ih = mdata.Data["IH"]

    Vb = mdata.Data["Ucoil15"] + mdata.Data["Ucoil16"]
    Ib = mdata.Data["IB"]
    timestep = 1  # for txt

if f_extension == ".tdms":
    # Uh = mdata.Data["Tensions_Aimant"]["ALL_internes"]
    if "ALL_internes" in mdata.Data["Tensions_Aimant"]:
        Vh = mdata.Data["Tensions_Aimant"][["ALL_internes", "t"]]
        Vh = Vh.set_index("t")
        print(f"Vh: {Vh.info(verbose=True)}, {Vh.head()}, {Vh.describe()}", flush=True)
        timestep = mdata.Groups["Tensions_Aimant"]["ALL_internes"]["wf_increment"]
    else:
        Vh = mdata.Data["Tensions_Aimant"][["ALL internes", "t"]]
        Vh = Vh.set_index("t")
        Vh.rename(columns={"ALL internes": "ALL_internes"}, inplace=True)
        timestep = mdata.Groups["Tensions_Aimant"]["ALL internes"]["wf_increment"]

    # print(f"Uh({t})=", end="", flush=True)
    if "Courant_GR1" in mdata.Data["Courants_Alimentations"]:
        Ih = mdata.Data["Courants_Alimentations"][["Courant_GR1", "t"]]
        Ih = Ih.set_index("t")
    else:
        Ih = mdata.Data["Courants_Alimentations"][["Courant GR1", "t"]]
        Ih = Ih.set_index("t")
        Ih.rename(columns={"Courant GR1": "Courant_GR1"}, inplace=True)

    if "ALL_externes" in mdata.Data["Tensions_Aimant"]:
        Vb = mdata.Data["Tensions_Aimant"][["ALL_externes", "t"]]
        Vb = Vb.set_index("t")
        timestep = mdata.Groups["Tensions_Aimant"]["ALL_externes"]["wf_increment"]
    else:
        Vb = mdata.Data["Tensions_Aimant"][["ALL externes", "t"]]
        Vb = Vb.set_index("t", "ALL externes")
        Vb.rename(columns={"ALL externes": "ALL_externes"}, inplace=True)
        timestep = mdata.Groups["Tensions_Aimant"]["ALL externes"]["wf_increment"]

    if "Courant_GR1" in mdata.Data["Courants_Alimentations"]:
        Ib = mdata.Data["Courants_Alimentations"][["Courant_GR2", "t"]]
        Ib = Ib.set_index("t")
    else:
        Ib = mdata.Data["Courants_Alimentations"][["Courant GR2", "t"]]
        Ib = Ib.set_index("t")
        Ib.rename(columns={"Courant GR2": "Courant_GR2"}, inplace=True)

    # print(f"Uh({t})=", end="", flush=True)
    if "Courant_GR1" in mdata.Data["Courants_Alimentations"]:
        Ih = mdata.Data["Courants_Alimentations"][["Courant_GR1", "t"]]
        Ih = Ih.set_index("t")
    else:
        Ih = mdata.Data["Courants_Alimentations"][["Courant GR1", "t"]]
        Ih = Ih.set_index("t")
        Ih.rename(columns={"Courant GR1": "Courant_GR1"}, inplace=True)

# Example data
n = Ih.to_numpy().size
print(f"samples: {n}", flush=True)
print(f"timestep: {timestep}", flush=True)

# Calculate the Nyquist frequency
nyquist_freq = (1 / timestep) / 2
print("Nyquist Frequency:", nyquist_freq, "Hz")

# force reshape to (size,)
data = {
    "X": np.reshape(Ih.to_numpy(), (n,)),
    "Y": np.reshape(Ib.to_numpy(), (n,)),
    "Z": np.reshape(Vh.to_numpy(), (n,)),
}

fft_data = {}
for key, value in data.items():
    print(f"key={key}", flush=True)
    print(f"raw data: type={type(value)}")
    print(f"size={value.size}")
    print(f"shape={value.shape}")

    vfft = fftshift(fft(value))
    print(type(vfft))
    print(vfft.size)
    freq = fftshift(fftfreq(vfft.size, d=timestep))
    # fft = pyfftw.interfaces.numpy_fft.fft(value)
    # freq = np.fft.fftfreq(n, d=1 / timestep)
    print(f"fft[{key}]:", vfft)
    fft_data[key] = vfft

# time_points = Vh.index.values
plt.plot(Vh.index.values, Vh.to_numpy(), label="Vh")
plt.legend()
plt.grid()
plt.xlabel("t [s]")
plt.show()

Rh = 0.0136
plt.plot(freq, fft_data["Z"].real, label="FFT Ucoil real")
plt.plot(
    freq, Rh * fft_data["X"].real, marker=".", linestyle="None", label="FFT R*Ih real"
)
plt.xlim(0.001, nyquist_freq)  # Only plot up to the Nyquist frequency
plt.legend()
plt.grid()
plt.xlabel("freq [Hz]")
plt.show()

Lh = 0.002315
M = 0.001965
imag_U = Lh * fft_data["X"].real + M * fft_data["Y"].real
for i, y in enumerate(np.nditer(imag_U, op_flags=["readwrite"])):
    y *= 2 * pi * freq[i]
imag_U += Rh * fft_data["X"].imag

plt.plot(freq, fft_data["Z"].imag, label="FFT Ucoil1 imag")
plt.plot(
    freq,
    imag_U,
    marker=".",
    linestyle="None",
    label="FFT w*(L*Ih + M*Ib) imag",
)
plt.xlim(0.001, nyquist_freq)  # Only plot up to the Nyquist frequency
plt.legend()
plt.grid()
plt.xlabel("freq [Hz]")
plt.show()

# Create a DataFrame
df = pd.DataFrame({"U": fft_data["Z"].real, "Ih": fft_data["X"].real})

# Define the independent variables (X and Y)
X = df[["Ih"]]

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Define the dependent variable (Z)
Y = df["U"]

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Get the summary of the model
print(model.summary())

# Extract the coefficients
intercept, A = model.params
print(f"Intercept: {intercept}, A: {A}")

# Create a DataFrame
df = pd.DataFrame(
    {
        "U": fft_data["Z"].imag,
        "iIh": fft_data["X"].imag,
        "rIh": fft_data["X"].real,
        # "rIb": fft_data["Y"].real,
    }
)

# Define the independent variables (X and Y)
# X = df[["iIh", "rIh", "rIb"]]
X = df[["iIh", "rIh"]]

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Define the dependent variable (Z)
Y = df["U"]

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Get the summary of the model
print(model.summary())

# Extract the coefficients
intercept, A, B = model.params
print(f"Intercept: {intercept}, A: {A}, B: {B}")
