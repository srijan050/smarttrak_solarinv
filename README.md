# SmartTrak_SolarInv

This repository contains the modeling and simulation of a solar inverter system using Physics-Informed Neural Networks (PINNs) for SmartTrak AI. The main objective is to integrate physical laws into neural network models to achieve robust predictions, especially in scenarios involving sensor faults or limited data availability.

## Table of Contents

- [Introduction](#introduction)
- [Physics-Informed Neural Networks (PINNs)](#physics-informed-neural-networks-pinns)
  - [Example: Ohm's Law in a Neural Network](#example-ohms-law-in-a-neural-network)
- [Solar Inverter Modeling](#solar-inverter-modeling)
  - [System Components](#system-components)
  - [Working Principle](#working-principle)
- [Implementation Overview](#implementation-overview)
  - [1. `iv_char.py` - Implementation of I-V Characteristics](#1-iv_charpy---implementation-of-i-v-characteristics)
  - [2. `mppt.py` - Implementation of MPPT Algorithm](#2-mpptpy---implementation-of-mppt-algorithm)
  - [3. `dcdc_conv.py` - DC-DC Converter Modeling](#3-dcdc_convpy---dc-dc-converter-modeling)
  - [4. `igbt_emi_filter.py` - IGBT Control and EMI Filter](#4-igbt_emi_filterpy---igbt-control-and-emi-filter)
- [Parameters Used](#parameters-used)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [Files Description](#files-description)
  - [1. `iv_char.py` - Implementation of I-V Characteristics](#1-iv_charpy---implementation-of-i-v-characteristics-1)
  - [2. `mppt.py` - Implementation of MPPT Algorithm](#2-mpptpy---implementation-of-mppt-algorithm-1)
  - [3. `dcdc_conv.py` - DC-DC Converter Modeling](#3-dcdc_convpy---dc-dc-converter-modeling-1)
  - [4. `igbt_emi_filter.py` - IGBT Control and EMI Filter](#4-igbt_emi_filterpy---igbt-control-and-emi-filter-1)
- [Instructions to Run the Code](#instructions-to-run-the-code)
- [Acknowledgments](#acknowledgments)

---

## Introduction

Solar inverters are critical components in photovoltaic (PV) systems, converting the DC power generated by solar panels into AC power suitable for use in homes or feeding into the grid. Modeling these inverters accurately is essential for optimizing performance, predicting behavior under various conditions, and integrating advanced control strategies like Physics-Informed Neural Networks (PINNs).

This repository presents a comprehensive model of a solar inverter system, including the PV panel's I-V characteristics, Maximum Power Point Tracking (MPPT) algorithm, DC-DC converter, IGBT-based inverter, and EMI filter. Each component is implemented in Python, with emphasis on physical accuracy and suitability for PINN integration.

---

## Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks incorporate the physical laws governing a system into the neural network's loss function. Alongside the traditional data loss (difference between predicted and actual values), a **physics loss** is included to enforce physical constraints.

### Example: Ohm's Law in a Neural Network

Consider a neural network predicting current (`I`) given voltage (`V`) and resistance (`R`). Traditional training would adjust weights and biases using sensor data but might ignore fundamental physical laws like Ohm's law (`V = IR`).

In a PINN, we introduce an additional loss based on the physics:

- **Physics Loss:** `V - IR` (should ideally be zero).
- **Total Loss:** `a * data_loss + b * physics_loss`, where `0 <= a, b <= 1` and `a + b = 1`.

This approach ensures the network not only fits the data but also adheres to physical laws, enhancing reliability, especially when sensor data is faulty or incomplete.

---

## Solar Inverter Modeling

### System Components

The solar inverter model consists of the following components:

1. **PV Panel I-V Characteristics:** Modeling the current-voltage relationship of the solar panel.
2. **MPPT Control:** Algorithm to find the point on the PV panel's I-V curve that maximizes power output.
3. **DC-DC Converter:** Steps up/down the DC voltage to the desired level.
4. **IGBT Control:** Uses Insulated Gate Bipolar Transistors (IGBTs) to convert DC to AC through high-frequency switching.
5. **EMI Filter:** Smooths the inverter output to produce a sinusoidal AC voltage suitable for use by appliances or feeding into the grid.

### Working Principle

- **DC Generation:** Solar PV panels generate DC power when sunlight excites electrons, creating a current flow.
- **MPPT:** The MPPT algorithm adjusts the operating point to extract maximum power from the PV panels.
- **Voltage Conversion:** The DC-DC converter adjusts the PV voltage to match the inverter's input requirements.
- **DC to AC Conversion:** The inverter uses IGBTs and Pulse Width Modulation (PWM) techniques to convert DC to AC.
- **Filtering:** An EMI filter attenuates high-frequency components, resulting in a clean AC output.

---

## Implementation Overview

This repository contains four Python files, each representing a component of the solar inverter system:

### 1. `iv_char.py` - Implementation of I-V Characteristics

#### Description

- **Purpose:** Models the I-V characteristics of a solar PV panel based on the single-diode model.
- **Functionality:** Calculates the current output (`I_pv`) of the PV panel for a given voltage (`V_pv`) considering factors like irradiance (`G`), temperature (`T`), and panel parameters.

#### Key Parameters

- **Photocurrent (`I_ph`):** Current generated by the photovoltaic effect, proportional to irradiance.
- **Saturation Current (`I_0`):** Reverse saturation current of the diode.
- **Series Resistance (`R_s`):** Represents resistive losses within the panel.
- **Shunt Resistance (`R_sh`):** Accounts for leakage currents.
- **Ideality Factor (`n`):** Diode ideality factor, affecting the diode equation.

#### Implementation Details

- **Equation: PV Panel Current**

  $$
  I_{\text{pv}} = I_{\text{ph}} - I_0 \left( e^{\frac{V_{\text{pv}} + I_{\text{pv}} R_s}{n V_t}} - 1 \right) - \frac{V_{\text{pv}} + I_{\text{pv}} R_s}{R_{\text{sh}}}
  $$

- **Solution Approach:** Uses iterative methods (e.g., Newton-Raphson) to solve for `I_pv` at each `V_pv`.

### 2. `mppt.py` - Implementation of MPPT Algorithm

#### Description

- **Purpose:** Implements the Perturb and Observe (P&O) MPPT algorithm to find the Maximum Power Point (MPP) of the PV panel.
- **Functionality:** Adjusts the operating voltage (`V_pv`) to maximize the power output (`P_pv`).

#### Key Parameters

- **Step Size (`delta_v`):** The voltage increment/decrement applied at each iteration.
- **Sampling Time (`dt`):** Time interval between successive measurements.

#### Implementation Details

- **Algorithm Steps:**
  1. Measure current power `P_current`.
  2. Perturb `V_pv` by `delta_v` and measure new power `P_new`.
  3. Compare `P_new` with `P_current`:
     - If `P_new > P_current`, continue perturbing in the same direction.
     - If `P_new < P_current`, reverse the perturbation direction.
  4. Repeat until convergence to MPP.

### 3. `dcdc_conv.py` - DC-DC Converter Modeling

#### Description

- **Purpose:** Models a DC-DC Boost Converter that steps up the PV voltage to a higher DC voltage required by the inverter.
- **Functionality:** Simulates the inductor current (`i_L`) and output voltage (`v_o`) over time, considering switching actions.

#### Key Parameters

- **Inductance (`L`):** Inductor value affecting current ripple.
- **Capacitance (`C`):** Output capacitor value affecting voltage ripple.
- **Load Resistance (`R_load`):** Represents the connected load.
- **Switching Frequency (`f_s`):** Frequency at which the converter's switch operates.
- **Duty Cycle (`D`):** Ratio of the ON time to the total switching period.

#### Implementation Details

- **Operating Phases:**
  - **Switch ON:** Inductor stores energy; current increases.
  - **Switch OFF:** Inductor releases energy to the output; current decreases.
- **Key Equations:**

  **Equation: Inductor Voltage**

  $$V_L = L \frac{di_L}{dt}$$

  **Equation: Capacitor Current**

  $$i_C = C \frac{dv_o}{dt}$$

- **Simulation Loop:**
  - Determines switch state based on the duty cycle.
  - Updates inductor current and capacitor voltage accordingly.
  - Ensures physical accuracy by starting with `v_o[0] = 0` to capture transient behavior.

### 4. `igbt_emi_filter.py` - IGBT Control and EMI Filter

#### Description

- **Purpose:** Models the inverter stage using IGBTs controlled by SPWM and implements an EMI filter to produce a sinusoidal output.
- **Functionality:**
  - Generates PWM signals using Sinusoidal Pulse Width Modulation (SPWM).
  - Simulates the inverter output voltage (`v_inverter`) and filters it using an LC EMI filter.

#### Key Parameters

- **IGBT Parameters:**
  - **Switching Frequency (`f_carrier`):** Frequency of the triangular carrier wave.
  - **Output Frequency (`f_out`):** Desired AC output frequency.
- **EMI Filter Components:**
  - **Filter Inductance (`L_f`):** Impedes high-frequency currents.
  - **Filter Capacitance (`C_f`):** Diverts high-frequency voltages.
- **Load Resistance (`R_load`):** Represents the connected AC load.

#### Implementation Details

- **SPWM Generation:**
  - **Reference Signal:** Low-frequency sine wave (`f_out`).
  - **Carrier Signal:** High-frequency triangular wave (`f_carrier`).
  - **PWM Signal Generation:** Compares the reference and carrier signals to produce gate signals for the IGBTs.
- **Inverter Modeling:**
  - Uses an H-bridge configuration with four IGBTs.
  - Switches the IGBTs to produce a PWM voltage switching between `+V_DC` and `-V_DC`.
- **EMI Filter Modeling:**

  **Equation: Inductor Current in EMI Filter**

  $$\frac{di_f}{dt} = \frac{v_{\text{inverter}} - v_{\text{filter}}}{L_f}$$

  **Equation: Capacitor Voltage in EMI Filter**

  $$\frac{dv_{\text{filter}}}{dt} = \frac{i_f - i_{\text{load}}}{C_f}$$

- **Simulation Loop:**
  - Updates inductor current and capacitor voltage using numerical methods.
  - Ensures the filtered output voltage (`v_filter`) is sinusoidal.

---

## Parameters Used

Below is a consolidated list of parameters used across the models:

### General Parameters

- **Sampling Times (`dt`, `dt_inverter`):** Time steps for numerical simulation.
- **Total Simulation Times (`t_total`, `t_total_inverter`):** Duration of the simulations.

### DC-DC Converter Parameters

- **Input Voltage (`V_pv_mpp`):** MPP voltage from the PV panel (e.g., 18 V).
- **Desired Output Voltage (`V_out_desired`):** Target voltage after boosting (e.g., 48 V).
- **Inductance (`L`):** Typically around 1 mH.
- **Capacitance (`C`):** Typically around 100 μF.
- **Load Resistance (`R_load`):** Adjusted based on desired output current.
- **Switching Frequency (`f_s`):** Commonly 50 kHz.
- **Duty Cycle (`D`):** Calculated based on input and output voltages.

### Inverter and EMI Filter Parameters

- **DC Input Voltage (`V_DC`):** Output voltage from the DC-DC converter.
- **Output Frequency (`f_out`):** Desired AC frequency (e.g., 50 Hz).
- **Carrier Frequency (`f_carrier`):** PWM switching frequency (e.g., 10 kHz).
- **Filter Inductance (`L_f`):** Typically around 1 mH.
- **Filter Capacitance (`C_f`):** Typically around 0.1 μF.
- **Load Resistance (`R_load`):** Represents the AC load (e.g., 10 Ω).

---

## Conclusion and Future Work

This repository provides a physics-based modeling approach to simulate a solar inverter system, integrating key components and control strategies. By utilizing Physics-Informed Neural Networks, the models ensure adherence to physical laws, enhancing reliability and robustness.

### Future Improvements

- **Advanced Control Strategies:**
  - Implement closed-loop control for the inverter.
  - Explore different MPPT algorithms.
- **Grid Integration:**
  - Model grid-connected scenarios with appropriate synchronization.
- **Validation:**
  - Compare simulation results with experimental data for model validation.

---

## Files Description

### 1. `iv_char.py` - Implementation of I-V Characteristics

- **Purpose:** Simulates the current-voltage characteristics of a solar PV panel.
- **Features:**
  - Models the PV panel based on the single-diode equation.
  - Calculates `I_pv` for a range of `V_pv` values.
  - Allows adjustment of irradiance and temperature to observe effects on the I-V curve.

### 2. `mppt.py` - Implementation of MPPT Algorithm

- **Purpose:** Implements the Perturb and Observe (P&O) algorithm to track the Maximum Power Point.
- **Features:**
  - Uses the I-V characteristics from `iv_char.py`.
  - Adjusts the operating voltage to maximize power output.
  - Plots power vs. voltage to visualize the MPP.

### 3. `dcdc_conv.py` - DC-DC Converter Modeling

- **Purpose:** Simulates a boost converter stepping up the PV voltage to a higher DC voltage.
- **Features:**
  - Models inductor current and output voltage over time.
  - Includes switching actions based on a calculated duty cycle.
  - Starts with initial conditions to capture transient behavior.

### 4. `igbt_emi_filter.py` - IGBT Control and EMI Filter

- **Purpose:** Simulates the inverter stage using IGBTs and implements an EMI filter.
- **Features:**
  - Generates SPWM signals for IGBT control.
  - Models the inverter output voltage and filters it to produce a sinusoidal waveform.
  - Includes visualization of reference and carrier signals, PWM output, and filtered output.

---

## Instructions to Run the Code

1. **Ensure Dependencies are Installed:**

   - `numpy`
   - `matplotlib`
   - `scipy` (if performing FFT analysis)

   Install them using:

   ```bash
   pip install numpy matplotlib scipy
   ```

2. **Execution Order:**

   - Run `iv_char.py` to generate the I-V characteristics.
   - Run `mppt.py` to perform MPPT and obtain `V_pv_mpp` and `I_pv_mpp`.
   - Update `dcdc_conv.py` with the MPP values from `mppt.py` and run it.
   - Use the output voltage from `dcdc_conv.py` in `igbt_emi_filter.py` and run it to simulate the inverter and EMI filter.

3. **Adjust Parameters as Needed:**

   - Modify component values and simulation settings to match specific scenarios or to observe different behaviors.

---

## Acknowledgments

- **SmartTrak AI:** For providing the opportunity to work on this project.

---

*This README was last updated on [Oct 6 2024].*
