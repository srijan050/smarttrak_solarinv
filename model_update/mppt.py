import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from iv_char import pv_current

# MPPT Algorithm Implementation
def mppt_po(initial_voltage, G, T, delta_V=0.01, num_iterations=100):
    """
    Implement the Perturb and Observe (P&O) MPPT algorithm.

    Parameters:
    - initial_voltage: Starting voltage (V)
    - G: Irradiance (W/m^2)
    - T: Temperature (K)
    - delta_V: Voltage perturbation step (V)
    - num_iterations: Number of iterations to simulate

    Returns:
    - V_history: List of voltages at each iteration
    - I_history: List of currents at each iteration
    - P_history: List of powers at each iteration
    """
    V = initial_voltage
    I = pv_current(V, G, T)
    P = V * I

    V_history = [V]
    I_history = [I]
    P_history = [P]

    direction = 1  # 1 for increasing voltage, -1 for decreasing voltage

    for _ in range(num_iterations):
        # Perturb voltage
        V_new = V + direction * delta_V
        I_new = pv_current(V_new, G, T)
        P_new = V_new * I_new

        # Compare power
        if P_new > P:
            # Power increased, keep the same direction
            V = V_new
            I = I_new
            P = P_new
        else:
            # Power decreased, reverse direction
            direction *= -1
            # Update with the new direction
            V_new = V + direction * delta_V
            I_new = pv_current(V_new, G, T)
            P_new = V_new * I_new
            V = V_new
            I = I_new
            P = P_new

        # Record the values
        V_history.append(V)
        I_history.append(I)
        P_history.append(P)

    return V_history, I_history, P_history

# Simulation parameters
G = 1000       # Irradiance (W/m^2)
T = 298.15     # Temperature (K)
initial_voltage = 0.5  # Initial voltage (V)
delta_V = 0.005        # Voltage perturbation step (V)
num_iterations = 100   # Number of iterations

# Run MPPT simulation
V_history, I_history, P_history = mppt_po(initial_voltage, G, T, delta_V, num_iterations)

# Plotting the results
plt.figure(figsize=(12, 6))

# Voltage vs. Iterations
plt.subplot(2, 2, 1)
plt.plot(V_history, label='Voltage (V)')
plt.title('Voltage during MPPT')
plt.xlabel('Iteration')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Current vs. Iterations
plt.subplot(2, 2, 2)
plt.plot(I_history, label='Current (A)', color='orange')
plt.title('Current during MPPT')
plt.xlabel('Iteration')
plt.ylabel('Current (A)')
plt.grid(True)
plt.legend()

# Power vs. Iterations
plt.subplot(2, 2, 3)
plt.plot(P_history, label='Power (W)', color='green')
plt.title('Power during MPPT')
plt.xlabel('Iteration')
plt.ylabel('Power (W)')
plt.grid(True)
plt.legend()

# I-V Curve with MPP Point
V_array = np.linspace(0, 0.7, 200)
I_array = [pv_current(V, G, T) for V in V_array]
P_array = V_array * I_array

plt.subplot(2, 2, 4)
plt.plot(V_array, I_array, label='I-V Curve')
plt.plot(V_history[-1], I_history[-1], 'ro', label='MPP')
plt.title('I-V Curve with MPP')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Print the final MPP values
print(f"Final MPP Voltage: {V_history[-1]:.4f} V")
print(f"Final MPP Current: {I_history[-1]:.4f} A")
print(f"Final MPP Power:   {P_history[-1]:.4f} W")