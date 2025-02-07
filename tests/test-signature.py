import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from python_magnetrun.MagnetRun import MagnetRun

def zscore(x, window):
    r = x.rolling(window=window)
    m = r.median().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

def piecewise_linear_approximation(serie, threshold: float=0.1):
    """
    Perform piecewise linear approximation on a time series and assign a signature.

    Parameters:
    - df: pandas DataFrame with a 'value' column containing the time series data.
    - threshold: float, the threshold to account for signal noise.

    Returns:
    - df: pandas DataFrame with an additional 'signature' column.
    """

    # Initialize variables
    signature = []
    previous_value = serie.iloc[0]
    previous_trend = None

    # Iterate over the dataframe
    for i in range(1, len(serie)):
        current_value = serie.iloc[i]
        difference = current_value - previous_value

        if difference > threshold:
            current_trend = "U"
        elif difference < -threshold:
            current_trend = "D"
        else:
            current_trend = "P"
        # print(i, current_value, difference, threshold, current_trend)

        # Append the trend to the signature list
        signature.append(current_trend)

        # Update previous values
        previous_value = current_value
        previous_trend = current_trend

    # Add the signature column to the dataframe
    signature.append(previous_trend)
    

    return signature


parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="enter input file")
parser.add_argument(
    "--site", help="specify a site (ex. M8, M9,...)", default="M9"
)  # use housing instead
parser.add_argument(
    "--threshold", help="set a threshold (default 0.5)", type=float, default=1.e-3
)  # use housing instead
parser.add_argument(
    "--window", help="set a window", type=int, default=10
)  # use housing instead
parser.add_argument("--save", help="activate plot", action="store_true")
args = parser.parse_args()
print(f"args: {args}", flush=True)

file = args.input_file
filename = os.path.basename(file)
index = filename.index("_")
site = filename[0:index]
print(f"site detected: {site}")

insert = "tututu"

mrun = MagnetRun.fromtxt(site, insert, file)
mdata = mrun.getMData()

key = "Field"
df = mdata.getData(key)
stats = df[key].describe()
print(f'stats for {key}:')
describe_list = stats.reset_index().to_dict(orient='records')
print(tabulate(describe_list, headers='keys', tablefmt='psql'))

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df[key], model='additive', period=args.window)
fig = result.plot()
# Add grid to all subplots
fig.suptitle(f"Seasonal Decomposition of {file}")
for ax in fig.axes:
    ax.grid(True)

if args.save:
    f_extension = os.path.splitext(file)[-1]
    imagefile = f"{file.replace(f_extension,'')}-{key}"
    plt.savefig(f"{imagefile}_seasonal_decompose.png", dpi=300)
    plt.close()
else:
    plt.show()
plt.close()


# Perform piecewise linear approximation
signature = piecewise_linear_approximation(result.trend, args.threshold)

# get index of changes in signature
changes = [0]
for i in range(1, len(signature)):
    if signature[i] != signature[i-1]:
        changes.append(i)
print(f"Signature: {[signature[i] for i in changes]}")
print(f"t: {[df.index[i] for i in changes]}")

# plot
my_ax = plt.gca()
legends = [key]
mdata.plotData(x="t", y=key, ax=my_ax, normalize=False)

legends.append('Trend')
result.trend.plot(ax=my_ax)
for x in [df.index[i] for i in changes]:
    plt.axvline(x=x, color="red")
my_ax.legend(labels=legends)
plt.grid()

if args.save:
    f_extension = os.path.splitext(file)[-1]
    imagefile = f"{file.replace(f_extension,'')}-{key}"
    plt.savefig(f"{imagefile}_signature.png", dpi=300)
    plt.close()
else:
    plt.show()
plt.close()

"""
# to get multiple figure
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

 
### s.rolling(window=5).mean()
# zscore(df[key], args.window).plot()

"""

