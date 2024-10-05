import numpy as np
import matplotlib.pyplot as plt

# Constants
q = 1.602176634e-19  # Electron charge (C)
k = 1.380649e-23     # Boltzmann's constant (J/K)

# Reference conditions
T_ref = 298.15       # Reference temperature (K)
G_ref = 1000         # Reference irradiance (W/m^2)

# PV Cell Parameters at Reference Conditions
I_ph_ref = 5.0       # Photocurrent at reference conditions (A)
I_s_ref = 1e-10      # Saturation current at reference conditions (A)
alpha_I = 0.0005     # Temperature coefficient of current (A/K)
E_g = 1.12           # Bandgap energy for silicon (eV)
n = 1.2              # Ideality factor (dimensionless)
R_s = 0.01           # Series resistance (Ohms)
R_sh = 1000          # Shunt resistance (Ohms)

# Functions to model parameters as variables
def photocurrent(G, T):
    """
    Calculate the photocurrent I_ph as a function of irradiance G and temperature T.
    """
    I_ph_T = I_ph_ref + alpha_I * (T - T_ref)
    I_ph = I_ph_T * (G / G_ref)
    return I_ph

def saturation_current(T):
    """
    Calculate the saturation current I_s as a function of temperature T.
    """
    T_ratio = T / T_ref
    I_s = I_s_ref * T_ratio**3 * np.exp((q * E_g) / (n * k) * (1 / T_ref - 1 / T))
    return I_s

def thermal_voltage(T):
    """
    Calculate the thermal voltage V_t as a function of temperature T.
    """
    V_t = (k * T) / q
    return V_t

def pv_current(V, G, T):
    """
    Calculate the output current I for a given voltage V, irradiance G, and temperature T.
    """
    I_ph = photocurrent(G, T)
    I_s = saturation_current(T)
    V_t = thermal_voltage(T)
    
    # Define the function to solve: I = I_ph - I_s * [exp((V + I*R_s)/(n*V_t)) - 1] - (V + I*R_s)/R_sh
    # Since I appears on both sides, we need to solve it numerically.
    # We can rearrange it as f(I) = 0 and find the root.
    def diode_eq(I):
        return I - I_ph + I_s * (np.exp((V + I * R_s) / (n * V_t)) - 1) + (V + I * R_s) / R_sh
    
    # Use Brent's method to solve for I
    from scipy.optimize import brentq
    I_min = -10  # Minimum current guess (A)
    I_max = I_ph  # Maximum current guess (A)
    try:
        I = brentq(diode_eq, I_min, I_max, xtol=1e-6)
    except ValueError:
        I = 0  # If no solution, set current to zero
    return I

# Simulation parameters
G = 1000       # Irradiance (W/m^2)
T = 298.15     # Temperature (K)
V_array = np.linspace(0, 0.6, 100)  # Voltage range (V)
I_array = []

# Calculate current for each voltage
for V in V_array:
    I = pv_current(V, G, T)
    I_array.append(I)

# Convert to numpy arrays
I_array = np.array(I_array)
P_array = V_array * I_array  # Power array

# Plotting the I-V and P-V curves
plt.figure(figsize=(10,5))

# I-V Curve
plt.subplot(1,2,1)
plt.plot(V_array, I_array, label='I-V Curve')
plt.title('I-V Characteristic of PV Cell')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.legend()

# P-V Curve
plt.subplot(1,2,2)
plt.plot(V_array, P_array, label='P-V Curve', color='orange')
plt.title('P-V Characteristic of PV Cell')
plt.xlabel('Voltage (V)')
plt.ylabel('Power (W)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
