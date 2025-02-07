import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from scipy.signal import correlate


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
    "--key", help="set key to consider", type=str, default="Field"
)
parser.add_argument(
    "--window", help="set a window", type=int, default=10
) 
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

# TODO get key symbol and unit from MagnetRun
key = args.key
df = mdata.getData(['t', key])
stats = df[key].describe()
print(f'stats for {key}:')
describe_list = stats.reset_index().to_dict(orient='records')
print(tabulate(describe_list, headers='keys', tablefmt='psql'))

# Perform piecewise linear approximation
# !! use index and not 't' column !!
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df[key], model='additive',period=args.window)
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


# Get signature in terms of Up, Plateau, Down
trend_component = result.trend
# Drop NaN values from trend slice
trend_component = trend_component[~np.isnan(trend_component)]
signature = piecewise_linear_approximation(trend_component, args.threshold)

# get index of changes in signature
changes = [0]
for i in range(1, len(signature)):
    if signature[i] != signature[i-1]:
        changes.append(i)
print(f"Signature: {[signature[i] for i in changes]}")
print(f"t: {[float(df['t'].iloc[i]) for i in changes]}")

# Plot piecewise linear approximation
my_ax = plt.gca()
legends = [key]
mdata.plotData(x="t", y=key, ax=my_ax, normalize=False)
for x in [df['t'].iloc[i] for i in changes]:
    plt.axvline(x=x, color="red")
my_ax.legend(labels=legends)
my_ax.grid(True)
plt.title(f"Piecewise Linear Approximation of {file}: {key}")

if args.save:
    f_extension = os.path.splitext(file)[-1]
    imagefile = f"{file.replace(f_extension,'')}-{key}"
    plt.savefig(f"{imagefile}_signature.png", dpi=300)
    plt.close()
else:
    plt.show()
plt.close()

#lag correlation between trend and field
# Compute cross-correlation

trend_component = result.trend
series = df[key].squeeze()

"""
# Select a slice of the time series
start_index = 2000
end_index = 2500
time_series_slice = series[start_index:end_index]
trend_slice = trend_component[start_index:end_index]

# Drop NaN values from trend slice
trend_slice = trend_slice[~np.isnan(trend_slice)]
time_series_slice = time_series_slice[-len(trend_slice):]

# Compute cross-correlation
correlation = correlate(time_series_slice - np.mean(time_series_slice), trend_slice - np.mean(trend_slice))
lags = np.arange(-len(correlation)//2 + 1, len(correlation)//2 + 1)

# Find the lag with maximum correlation
lag = lags[np.argmax(correlation)]

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_series_slice, label='Time Series Slice')
plt.plot(trend_slice, label='Trend Slice', color='red')
plt.title('Time Series Slice and Trend Slice')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(lags, correlation, label='Cross-Correlation')
plt.axvline(lag, color='red', linestyle='--', label=f'Lag = {lag}')
plt.title('Cross-Correlation between Time Series Slice and Trend Slice')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.legend()
plt.show()

print(f"Estimated lag: {lag} for {start_index} to {end_index}")
"""

"""
# to get multiple figure
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)

 
### s.rolling(window=5).mean()
# zscore(df[key], args.window).plot()

"""

