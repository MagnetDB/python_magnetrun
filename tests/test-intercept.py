"""
idea from chatgpt
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression

from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.processing.smoothers import savgol

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_file", help="enter input file (ex: ~/M9_Overview_240509-1634.tdms)"
)
parser.add_argument(
    "--window", help="stopping criteria for nlopt", type=int, default=10
)
args = parser.parse_args()
print(f"args: {args}", flush=True)

file = args.input_file
window = args.window

filename = os.path.basename(file)
site = filename.split("_")[0]
insert = "tutu"
mrun = MagnetRun.fromtdms(site, insert, file)
mdata = mrun.getMData()

# Helices insert

Uh = mdata.Data["Tensions_Aimant"]["ALL_internes"]
Ih = mdata.Data["Courants_Alimentations"]["Courant_GR1"]

Uh_smoothed = savgol(
    y=Uh.to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
print(f"Uh: {Uh_smoothed.shape}")
Ih_smoothed = savgol(
    y=Ih.to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
print(f"Ih: {Ih_smoothed.shape}")
dIh_smoothed = savgol(
    y=Ih.to_numpy(),
    window=window,
    polyorder=3,
    deriv=1,
)
print(f"dIh: {dIh_smoothed.shape}")

# Bitters
Ub = mdata.Data["Tensions_Aimant"]["ALL_externes"]
Ib = mdata.Data["Courants_Alimentations"]["Courant_GR2"]

Ub_smoothed = savgol(
    y=Ub.to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
print(f"Ub: {Ub_smoothed.shape}")
Ib_smoothed = savgol(
    y=Ib.to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
print(f"Ib: {Ib_smoothed.shape}")
dIb_smoothed = savgol(
    y=Ib.to_numpy(),
    window=window,
    polyorder=3,
    deriv=1,
)
print(f"dIb: {dIb_smoothed.shape}")

data = {
    "Uh": Uh_smoothed,
    "Ih": Ih_smoothed,
    "dIh": dIh_smoothed,
    "Ub": Ub_smoothed,
    "Ib": Ib_smoothed,
    "dIb": dIb_smoothed,
}

# Create a DataFrame
df = pd.DataFrame(data)
for key in data:
    print(f"{key}: contains NaN {df[key].isnull().any()}")

# Define the independent variables for each equation
X1 = df[["Ih", "dIh"]]
X2 = df[["Ib", "dIb"]]
print(X1.head())
print(X2.head())

print(f"X1 contains NaN:\n{X1.isnull().any()}")
print(f"X2 contains NaN:\n{X2.isnull().any()}")


# Define the dependent variables
Y1 = df["Uh"]
Y2 = df["Ub"]
print(f"Y1 contains NaN:\n{Y1.isnull().any()}")
print(f"Y2 contains NaN:\n{Y2.isnull().any()}")

# Concatenate the dependent variables and independent variables
X = pd.concat([X1, X2], axis=1)  # , axis=0)
print("X=", X.head())

"""
ax = plt.gca()
X.plot(ax=ax)
plt.show()
"""

Y = pd.concat([Y1, Y2], axis=1)
print("Y=", Y.head())
print(f"X contains NaN:\n{X.isnull().any()}")
print(f"Y contains NaN:\n{Y.isnull().any()}")

# Fit the regression model
model = LinearRegression().fit(X, Y)

# Extract the coefficients
res = model.coef_
intercept = model.intercept_
print(f"Intercept: {intercept}, res={res}")


[Rh, Lh, err, Mh] = res[0]
Y["estimated_Uh"] = Rh * X["Ih"] + Lh * X["dIh"] + err * X["Ib"] + Mh * X["dIb"]

[err, M, Rb, Lb] = res[1]
Y["estimated_Ub"] = Rb * X["Ib"] + Lb * X["dIb"] + err * X["Ih"] + M * X["dIh"]

ax = Y.plot()

plt.grid()
plt.legend()
plt.show()
