import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from iv_char import pv_current
from mppt import mppt_po

# Inverter Parameters
V_DC = 48           # DC input voltage from the DC-DC converter (V)
f_out = 50          # Desired output AC frequency (Hz)
f_carrier = 10e3    # Carrier frequency for PWM (Hz)
R_load = 10         # Load resistance (Ohms)

# Simulation Parameters
t_total = 0.001       # Total simulation time (s)
fs = 10 * f_carrier # Sampling frequency (Hz)
dt = 1 / fs         # Time step (s)
time = np.arange(0, t_total, dt)
time_steps = len(time)

# Generate Reference and Carrier Signals
ref_signal = np.sin(2 * np.pi * f_out * time)
carrier_signal = np.sign(np.sin(2 * np.pi * f_carrier * time))

# Initialize Gate Signals for IGBTs
gate_Q1 = np.zeros(time_steps)
gate_Q2 = np.zeros(time_steps)
gate_Q3 = np.zeros(time_steps)
gate_Q4 = np.zeros(time_steps)

# Generate PWM Signals
for t in range(time_steps):
    if ref_signal[t] >= carrier_signal[t]:
        gate_Q1[t] = 1
        gate_Q4[t] = 1
        gate_Q2[t] = 0
        gate_Q3[t] = 0
    else:
        gate_Q1[t] = 0
        gate_Q4[t] = 0
        gate_Q2[t] = 1
        gate_Q3[t] = 1

# Calculate Output Voltage and Current
v_o = np.zeros(time_steps)
i_o = np.zeros(time_steps)

for t in range(1, time_steps):
    # Determine output voltage based on IGBT states
    if gate_Q1[t] == 1 and gate_Q4[t] == 1:
        v_o[t] = V_DC
    elif gate_Q2[t] == 1 and gate_Q3[t] == 1:
        v_o[t] = -V_DC
    else:
        v_o[t] = 0  # Both switches off or invalid state (should not occur in SPWM)
    
    # Output current through resistive load
    i_o[t] = v_o[t] / R_load

# Plotting the results
plt.figure(figsize=(14, 10))

# Reference and Carrier Signals
plt.subplot(4, 1, 1)
plt.plot(time, ref_signal, label='Reference Signal')
plt.plot(time, carrier_signal, label='Carrier Signal', alpha=0.5)
plt.title('Reference and Carrier Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Gate Signals
plt.subplot(4, 1, 2)
plt.plot(time, gate_Q1, label='Gate Q1')
plt.plot(time, gate_Q2, label='Gate Q2')
plt.plot(time, gate_Q3, label='Gate Q3')
plt.plot(time, gate_Q4, label='Gate Q4')
plt.title('Gate Signals for IGBTs')
plt.xlabel('Time (s)')
plt.ylabel('Gate Voltage')
plt.grid(True)
plt.legend()

# Output Voltage
plt.subplot(4, 1, 3)
plt.plot(time, v_o, label='Output Voltage $v_o$', color='orange')
plt.title('Inverter Output Voltage (PWM)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Output Current
plt.subplot(4, 1, 4)
plt.plot(time, i_o, label='Output Current $i_o$', color='green')
plt.title('Output Current through Load')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
