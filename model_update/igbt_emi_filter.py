import numpy as np
import matplotlib.pyplot as plt

# Inverter Parameters
V_DC = 48           # DC input voltage from the DC-DC converter (V)
f_out = 50          # Desired output AC frequency (Hz)
f_carrier = 10e3    # Carrier frequency for PWM (Hz)
R_load = 10         # Load resistance (Ohms)

# EMI Filter Parameters
L_f = 2e-3          # Filter inductance (H)
C_f = 0.22e-6        # Filter capacitance (F)

# Calculate cut-off frequency
f_c = 1 / (2 * np.pi * np.sqrt(L_f * C_f))
print(f"EMI Filter Cut-off Frequency: {f_c:.1f} Hz")

# Simulation Parameters
t_total = 0.1       # Total simulation time (s)
fs = 100 * f_carrier  # Sampling frequency (Hz)
dt = 1 / fs         # Time step (s)
time = np.arange(0, t_total, dt)
time_steps = len(time)

# Generate Reference Signal (Sine Wave)
ref_signal = np.sin(2 * np.pi * f_out * time)

# Generate Triangular Carrier Signal
carrier_signal = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f_carrier * time))

# Initialize Gate Signals for IGBTs
gate_Q1 = np.zeros(time_steps)
gate_Q2 = np.zeros(time_steps)
gate_Q3 = np.zeros(time_steps)
gate_Q4 = np.zeros(time_steps)

# Generate PWM Signals using SPWM
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

# Calculate Inverter Output Voltage (before EMI filter)
v_inverter = np.zeros(time_steps)

for t in range(time_steps):
    if gate_Q1[t] == 1 and gate_Q4[t] == 1:
        v_inverter[t] = V_DC
    elif gate_Q2[t] == 1 and gate_Q3[t] == 1:
        v_inverter[t] = -V_DC
    else:
        v_inverter[t] = 0  # Both switches off or invalid state (should not occur in SPWM)

# Initialize EMI Filter Variables
i_f = np.zeros(time_steps)         # Inductor current through EMI filter
v_filter = np.zeros(time_steps)    # Voltage across filter capacitor (output voltage after filter)
i_load = np.zeros(time_steps)      # Load current

# Simulation Loop for EMI Filter
for t in range(1, time_steps):
    # Inductor voltage across L_f
    v_Lf = v_inverter[t-1] - v_filter[t-1]
    
    # Update inductor current
    di_f = (v_Lf / L_f) * dt
    i_f[t] = i_f[t-1] + di_f
    
    # Calculate load current
    i_load[t] = v_filter[t-1] / R_load
    
    # Update capacitor voltage
    dv_filter = ((i_f[t] - i_load[t]) / C_f) * dt
    v_filter[t] = v_filter[t-1] + dv_filter

# Plotting the results
plt.figure(figsize=(14, 12))

# Reference and Carrier Signals
plt.subplot(5, 1, 1)
plt.plot(time, ref_signal, label='Reference Signal (50 Hz)')
plt.plot(time, carrier_signal, label='Triangular Carrier Signal (10 kHz)', alpha=0.7)
plt.title('Reference and Carrier Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Gate Signals
plt.subplot(5, 1, 2)
plt.plot(time, gate_Q1, label='Gate Q1')
plt.plot(time, gate_Q2, label='Gate Q2')
plt.title('Gate Signals for IGBTs')
plt.xlabel('Time (s)')
plt.ylabel('Gate Voltage')
plt.grid(True)
plt.legend()

# Inverter Output Voltage (PWM)
plt.subplot(5, 1, 3)
plt.plot(time, v_inverter, label='Inverter Output Voltage $v_{inverter}$', color='orange')
plt.title('Inverter Output Voltage (PWM)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Filtered Output Voltage
plt.subplot(5, 1, 4)
plt.plot(time, v_filter, label='Filtered Output Voltage $v_{filter}$', color='blue')
plt.title('Filtered Output Voltage after EMI Filter')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Zoomed-In Filtered Output Voltage
plt.subplot(5, 1, 5)
start_zoom = int(0.02 * fs)
end_zoom = int(0.04 * fs)
plt.plot(time[start_zoom:end_zoom], v_filter[start_zoom:end_zoom], label='Filtered Output Voltage (Zoomed)', color='blue')
plt.title('Zoomed-In Filtered Output Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
