= lag correlation on slice

```python
from scipy.signal import correlate
import matplotlib.pyplot as plt

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
```

= multiple plots

```python
import matplotlib.pyplot as plt

# to get multiple figure
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Aligning x-axis using sharex')
ax1.plot(x, y)
ax2.plot(x + 1, -y)
```
 
= zscore

```python
def zscore(x, window):
    r = x.rolling(window=window)
    m = r.median().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z

### s.rolling(window=5).mean()
zscore(df[key], args.window).plot()
```



