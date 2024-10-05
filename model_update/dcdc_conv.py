import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from iv_char import pv_current
from mppt import mppt_po

# Simulation parameters
G = 1000       # Irradiance (W/m^2)
T = 298.15     # Temperature (K)
initial_voltage = 0.5  # Initial voltage (V)
delta_V = 0.005        # Voltage perturbation step (V)
num_iterations = 100   # Number of iterations

# Run MPPT simulation
V_history, I_history, P_history = mppt_po(initial_voltage, G, T, delta_V, num_iterations)

def calculate_duty_cycle(V_in, V_out_desired):
    """
    Calculate the duty cycle D required for a boost converter to achieve the desired output voltage.
    
    V_in: Input voltage (V)
    V_out_desired: Desired output voltage (V)
    """
    D = 1 - (V_in / V_out_desired)
    return D

# Use the final MPP voltage and current from MPPT
V_pv_mpp = V_history[-1]  # MPP voltage from MPPT
I_pv_mpp = I_history[-1]  # MPP current from MPPT

V_in = V_pv_mpp  # Input voltage is the PV panel voltage at MPP

# Desired Output Voltage (e.g., charging a 48V battery)
V_out_desired = 5  # Desired output voltage (V)

# Calculate the duty cycle required
D = calculate_duty_cycle(V_in, V_out_desired)
print(f"Calculated Duty Cycle for Desired Output Voltage: {D:.2f}")

# Update the Boost Converter simulation parameters with the new V_in and D

# Boost Converter Parameters (same as before)
L = 1e-3          # Inductance (H)
C = 1e-4          # Capacitance (F)
R_load = V_out_desired / ((V_in * I_pv_mpp) / V_out_desired)  # Adjust load to match power
f_s = 50e3        # Switching frequency (Hz)
T_s = 1 / f_s     # Switching period (s)

# Simulation Parameters (same as before)
t_total = 5    # Total simulation time (s)
dt = T_s / 100    # Time step (s)
time_steps = int(t_total / dt)
time = np.linspace(0, t_total, time_steps)

# Initialize variables (same as before)
i_L = np.zeros(time_steps)
v_o = np.zeros(time_steps)
i_o = np.zeros(time_steps)
state = np.zeros(time_steps)

# Initial Conditions
i_L[0] = 0
v_o[0] = 0

# Simulation Loop (same as before)
for t in range(1, time_steps):
    # Determine switch state
    t_current = time[t] % T_s
    if t_current < D * T_s:
        # Switch is ON
        switch_on = True
        state[t] = 1
        v_L = V_in
        i_diode = 0
    else:
        # Switch is OFF
        switch_on = False
        state[t] = 0
        v_L = V_in - v_o[t-1]
        i_diode = i_L[t-1]
    
    # Inductor current update
    di_L = (v_L / L) * dt
    i_L[t] = i_L[t-1] + di_L
    
    # Output current and voltage update
    i_o[t] = v_o[t-1] / R_load
    dv_o = ((i_diode - i_o[t]) / C) * dt
    v_o[t] = v_o[t-1] + dv_o

# Plotting and results (same as before)
plt.figure(figsize=(12, 8))

# Inductor Current
plt.subplot(3, 1, 1)
plt.plot(time * 1000, i_L, label='Inductor Current $i_L$')
plt.title('Inductor Current')
plt.xlabel('Time (ms)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.legend()

# Output Voltage
plt.subplot(3, 1, 2)
plt.plot(time * 1000, v_o, label='Output Voltage $v_o$', color='orange')
plt.title('Output Voltage')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Switch State
plt.subplot(3, 1, 3)
plt.plot(time * 1000, state, label='Switch State', color='green')
plt.title('Switch State')
plt.xlabel('Time (ms)')
plt.ylabel('State')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Print final values
print(f"Final Output Voltage: {v_o[-1]:.2f} V")
print(f"Final Inductor Current: {i_L[-1]:.2f} A")
print(f"Final Output Current: {i_o[-1]:.2f} A")
