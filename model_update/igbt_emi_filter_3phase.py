import numpy as np
import matplotlib.pyplot as plt

# Inverter Parameters
V_DC = 48            # DC input voltage from the DC-DC converter (V)
f_out = 50           # Desired output AC frequency (Hz)
f_carrier = 10e3     # Carrier frequency for PWM (Hz)
R_load = 10          # Load resistance per phase (Ohms)

# EMI Filter Parameters (per phase)
L_f = 2e-3           # Filter inductance (H)
C_f = 0.22e-6        # Filter capacitance (F)

# Calculate cut-off frequency
f_c = 1 / (2 * np.pi * np.sqrt(L_f * C_f))
print(f"EMI Filter Cut-off Frequency: {f_c:.1f} Hz")

# Simulation Parameters
t_total = 0.001        # Total simulation time (s)
fs = 100 * f_carrier  # Sampling frequency (Hz)
dt = 1 / fs          # Time step (s)
time = np.arange(0, t_total, dt)
time_steps = len(time)

# Generate Reference Signals (Sine Waves for Three Phases)
ref_signal_A = np.sin(2 * np.pi * f_out * time)
ref_signal_B = np.sin(2 * np.pi * f_out * time - (2/3)*np.pi)  # 120 degrees lagging
ref_signal_C = np.sin(2 * np.pi * f_out * time - (4/3)*np.pi)  # 240 degrees lagging

# Generate Triangular Carrier Signal
carrier_signal = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f_carrier * time))

# Initialize Gate Signals for IGBTs
gate_P1 = np.zeros(time_steps)
gate_P2 = np.zeros(time_steps)
gate_P3 = np.zeros(time_steps)
gate_N1 = np.zeros(time_steps)
gate_N2 = np.zeros(time_steps)
gate_N3 = np.zeros(time_steps)

# Generate PWM Signals using SPWM for Three Phases
for t in range(time_steps):
    # Phase A
    if ref_signal_A[t] >= carrier_signal[t]:
        gate_P1[t] = 1
        gate_N1[t] = 0
    else:
        gate_P1[t] = 0
        gate_N1[t] = 1
    # Phase B
    if ref_signal_B[t] >= carrier_signal[t]:
        gate_P2[t] = 1
        gate_N2[t] = 0
    else:
        gate_P2[t] = 0
        gate_N2[t] = 1
    # Phase C
    if ref_signal_C[t] >= carrier_signal[t]:
        gate_P3[t] = 1
        gate_N3[t] = 0
    else:
        gate_P3[t] = 0
        gate_N3[t] = 1

# Calculate Inverter Output Voltages (Line-to-Neutral Voltages)
v_AN = np.zeros(time_steps)
v_BN = np.zeros(time_steps)
v_CN = np.zeros(time_steps)

for t in range(time_steps):
    # DC Bus Voltage Allocation
    V_p = V_DC / 2  # Positive DC bus voltage
    V_n = -V_DC / 2  # Negative DC bus voltage

    # Phase A
    if gate_P1[t] == 1:
        v_AN[t] = V_p
    else:
        v_AN[t] = V_n

    # Phase B
    if gate_P2[t] == 1:
        v_BN[t] = V_p
    else:
        v_BN[t] = V_n

    # Phase C
    if gate_P3[t] == 1:
        v_CN[t] = V_p
    else:
        v_CN[t] = V_n

# Calculate Line-to-Line Voltages (Optional)
v_AB = v_AN - v_BN
v_BC = v_BN - v_CN
v_CA = v_CN - v_AN

# Initialize EMI Filter Variables for Each Phase
i_f_A = np.zeros(time_steps)
v_out_A = np.zeros(time_steps)
i_load_A = np.zeros(time_steps)

i_f_B = np.zeros(time_steps)
v_out_B = np.zeros(time_steps)
i_load_B = np.zeros(time_steps)

i_f_C = np.zeros(time_steps)
v_out_C = np.zeros(time_steps)
i_load_C = np.zeros(time_steps)

# Simulation Loop for EMI Filter
for t in range(1, time_steps):
    # Phase A
    v_Lf_A = v_AN[t-1] - v_out_A[t-1]
    di_f_A = (v_Lf_A / L_f) * dt
    i_f_A[t] = i_f_A[t-1] + di_f_A
    i_load_A[t] = v_out_A[t-1] / R_load
    dv_out_A = ((i_f_A[t] - i_load_A[t]) / C_f) * dt
    v_out_A[t] = v_out_A[t-1] + dv_out_A

    # Phase B
    v_Lf_B = v_BN[t-1] - v_out_B[t-1]
    di_f_B = (v_Lf_B / L_f) * dt
    i_f_B[t] = i_f_B[t-1] + di_f_B
    i_load_B[t] = v_out_B[t-1] / R_load
    dv_out_B = ((i_f_B[t] - i_load_B[t]) / C_f) * dt
    v_out_B[t] = v_out_B[t-1] + dv_out_B

    # Phase C
    v_Lf_C = v_CN[t-1] - v_out_C[t-1]
    di_f_C = (v_Lf_C / L_f) * dt
    i_f_C[t] = i_f_C[t-1] + di_f_C
    i_load_C[t] = v_out_C[t-1] / R_load
    dv_out_C = ((i_f_C[t] - i_load_C[t]) / C_f) * dt
    v_out_C[t] = v_out_C[t-1] + dv_out_C

# Plotting the results
plt.figure(figsize=(14, 16))

# Reference Signals
plt.subplot(6, 1, 1)
plt.plot(time, ref_signal_A, label='Reference Signal A (0°)')
plt.plot(time, ref_signal_B, label='Reference Signal B (-120°)')
plt.plot(time, ref_signal_C, label='Reference Signal C (-240°)')
plt.title('Reference Signals for Three Phases')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Carrier Signal
plt.subplot(6, 1, 2)
plt.plot(time, carrier_signal, label='Triangular Carrier Signal (10 kHz)', color='gray')
plt.title('Carrier Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Inverter Output Voltages (PWM)
plt.subplot(6, 1, 3)
plt.plot(time, v_AN, label='Inverter Output Voltage v_AN', color='orange')
plt.plot(time, v_BN, label='Inverter Output Voltage v_BN', color='green')
plt.plot(time, v_CN, label='Inverter Output Voltage v_CN', color='purple')
plt.title('Inverter Output Voltages (PWM)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Filtered Output Voltages
plt.subplot(6, 1, 4)
plt.plot(time, v_out_A, label='Filtered Output Voltage v_out_A', color='blue')
plt.plot(time, v_out_B, label='Filtered Output Voltage v_out_B', color='red')
plt.plot(time, v_out_C, label='Filtered Output Voltage v_out_C', color='brown')
plt.title('Filtered Output Voltages after EMI Filter')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Zoomed-In Filtered Output Voltages
plt.subplot(6, 1, 5)
start_zoom = int(0.02 * fs)
end_zoom = int(0.04 * fs)
plt.plot(time[start_zoom:end_zoom], v_out_A[start_zoom:end_zoom], label='v_out_A (Zoomed)', color='blue')
plt.plot(time[start_zoom:end_zoom], v_out_B[start_zoom:end_zoom], label='v_out_B (Zoomed)', color='red')
plt.plot(time[start_zoom:end_zoom], v_out_C[start_zoom:end_zoom], label='v_out_C (Zoomed)', color='brown')
plt.title('Zoomed-In Filtered Output Voltages')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

# Line-to-Line Voltages (Filtered)
v_out_AB = v_out_A - v_out_B
v_out_BC = v_out_B - v_out_C
v_out_CA = v_out_C - v_out_A

plt.subplot(6, 1, 6)
plt.plot(time, v_out_AB, label='Filtered Line Voltage v_out_AB', color='cyan')
plt.plot(time, v_out_BC, label='Filtered Line Voltage v_out_BC', color='magenta')
plt.plot(time, v_out_CA, label='Filtered Line Voltage v_out_CA', color='yellow')
plt.title('Filtered Line-to-Line Voltages')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
