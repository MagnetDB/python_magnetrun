import matplotlib.pyplot as plt

# Create a plot
fig, ax = plt.subplots()

# Add vertical spans with colors and alpha values
ax.axvspan(0.1, 0.4, color='blue', alpha=0.5)
ax.axvspan(0.3, 0.6, color='red', alpha=0.5)

# Add horizontal spans with colors and alpha values
ax.axhspan(0.3, 0.6, color='green', alpha=0.5)
ax.axhspan(0.5, 0.8, color='yellow', alpha=0.5)

# Set limits and display the plot
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.title("Alpha Blending with axvspan and axhspan")
plt.show()

