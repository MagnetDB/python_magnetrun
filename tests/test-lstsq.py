import numpy as np
from scipy.linalg import lstsq

# Example time points
time_points = np.linspace(0, 10, 100)  # Example time points

# Example currents (replace with actual measurements)
I1_measured = np.array(
    [np.sin(time_points), np.cos(time_points)]
).T  # Current for circuit 1
I2_measured = np.array(
    [np.sin(time_points + np.pi / 4), np.cos(time_points + np.pi / 4)]
).T  # Current for circuit 2

# Example voltages (replace with actual measurements)
U1_measured = np.array(
    [5 * np.sin(time_points), 10 * np.cos(time_points)]
).T  # Voltage for circuit 1
U2_measured = np.array(
    [5 * np.sin(time_points + np.pi / 4), 10 * np.cos(time_points + np.pi / 4)]
).T  # Voltage for circuit 2

# Compute the time derivatives of the currents
dI1_dt_measured = np.gradient(I1_measured, time_points, axis=0)
dI2_dt_measured = np.gradient(I2_measured, time_points, axis=0)

# Form the matrix A for both circuits
A1 = np.hstack([I1_measured, dI1_dt_measured])
A2 = np.hstack([I2_measured, dI2_dt_measured])
A_combined = np.vstack([A1, A2])

# Combine the voltage measurements
U_combined = np.vstack([U1_measured, U2_measured]).reshape(-1)

# Ensure A_combined has the correct dimensions
assert A_combined.shape == (2 * len(time_points), 4), "A_combine"
# Ensure U_combined has the correct dimension
assert U_combined.shape == (2 * 2 * len(time_points),), "U_combine"

# Solve the least squares problem
R_L_vector, _, _, _ = lstsq(A_combined, U_combined)

# Extract the R and L matrices from the solution vector
R = R_L_vector[:4].reshape((2, 2))
L = R_L_vector[4:].reshape((2, 2))

# Output the estimated matrices
print("Estimated Resistance Matrix R:")
print(R)
print("Estimated Inductance Matrix L:")
print(L)
