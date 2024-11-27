import piecewise_regression
import numpy as np

alpha_1 = -4
alpha_2 = -2
constant = 100
breakpoint_1 = 7
n_points = 200
np.random.seed(0)
xx = np.linspace(0, 20, n_points)
print(xx.shape)
print(type(xx))
yy = constant + alpha_1*xx + (alpha_2-alpha_1) * np.maximum(xx - breakpoint_1, 0) + np.random.normal(size=n_points)

# Given some data, fit the model
pw_fit = piecewise_regression.Fit(xx, yy, start_values=[5], n_breakpoints=1)

# Print a summary of the fit
pw_fit.summary()

import matplotlib.pyplot as plt

# Plot the data, fit, breakpoints and confidence intervals
pw_fit.plot_data(color="grey", s=20)
# Pass in standard matplotlib keywords to control any of the plots
pw_fit.plot_fit(color="red", linewidth=4)
pw_fit.plot_breakpoints()
pw_fit.plot_breakpoint_confidence_intervals()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()

