import pytest

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

rng = np.random.default_rng()
x = rng.standard_normal(1000)
y = np.concatenate([rng.standard_normal(100), x])
correlation = signal.correlate(x, y, mode="full")
lags = signal.correlation_lags(x.size, y.size, mode="full")
lag = lags[np.argmax(correlation)]
print("lag=", lag)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x, color="b")
plt.plot(y, color="r")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(lags, correlation, label="Cross-Correlation")
plt.axvline(lag, color="red", linestyle="--", label=f"Lag = {lag}")
plt.grid()

plt.show()
plt.close()
