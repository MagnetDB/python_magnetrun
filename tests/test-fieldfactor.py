"""
idea from chatgpt
"""

import argparse
import logging
import os

import pandas as pd
import statsmodels.api as sm

from python_magnetrun.cli_args import create_base_parser
from python_magnetrun.log_utils import setup_logging
from python_magnetrun.MagnetRun import MagnetRun
from python_magnetrun.processing.smoothers import savgol

_default_input = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "M9_2019.02.14---23_00_38.txt",
)

logger = logging.getLogger("python_magnetrun")
command_line = None
base_parser = create_base_parser(add_input_file=False)
parser = argparse.ArgumentParser("Field factor identification", parents=[base_parser])
parser.add_argument(
    "input_file",
    nargs="?",
    default=_default_input,
    help="enter input file (ex: ~/M9_2024.05.13---16_30_51.txt)",
)
parser.add_argument(
    "--window", help="stopping criteria for nlopt", type=int, default=10
)
args, _unknown = parser.parse_known_args()
if _unknown:
    parser.error(f"unrecognized arguments: {' '.join(_unknown)}")
setup_logging(level=args.log_level, log_file=args.log_file)
logger.debug(f"args: {args}")

file = args.input_file
window = args.window

filename = os.path.basename(file)
housing = filename.split("_")[0] if args.housing is None else args.housing
site = args.site
insert = args.insert
mrun = MagnetRun.fromtxt(housing, site, file)
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
