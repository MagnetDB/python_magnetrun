"""
idea from chatgpt
"""

import os
import pandas as pd
import statsmodels.api as sm

from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.processing.smoothers import savgol

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_file",
    help="enter input file (ex: ~/M9_2024.05.13---16_30_51.txt)",
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
mrun = MagnetRun.fromtxt(site, insert, file)
mdata = mrun.getMData()


B = mdata.Data["Field"]
Ih = mdata.Data["IH"]
Ib = mdata.Data["IB"]

B_smoothed = savgol(
    y=B.to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
Ih_smoothed = savgol(
    y=Ih.to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)
Ib_smoothed = savgol(
    y=Ib.to_numpy(),
    window=window,
    polyorder=3,
    deriv=0,
)

# Example data
# Suppose you have your time series data for X, Y, and Z
data = {
    "X": Ih_smoothed,
    "Y": Ib_smoothed,
    "Z": B_smoothed,
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the independent variables (X and Y)
X = df[["X", "Y"]]

# Add a constant to the model (the intercept)
X = sm.add_constant(X)

# Define the dependent variable (Z)
Y = df["Z"]

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Get the summary of the model
print(model.summary())

# Extract the coefficients
intercept, A, B = model.params
print(f"Intercept: {intercept}, A: {A}, B: {B}")
