"""
Title: Reactor Antineutrino Survival Probability with Scalar and Tensor NSI

Description:
This script computes and plots the ratio of survival probabilities for electron antineutrinos
at far and near detectors, including non-standard interactions (NSI)
of scalar and tensor type. The starting point is based on Equations (3.12) and (3.13) of the paper:
"Probing Non-Standard Neutrino Interactions with Reactor Neutrinos" (arXiv:1901.04553v3).

NSI parameters (Re[S], Im[S], Re[T], Im[T]) are set manually in the script.
The effect is visualized by plotting the far/near survival probability ratio
as a function of neutrino energy.

Author: Pietro Chimenti
Affiliation: Universidade Estadual de Londrina (UEL)
Date: 2025-05-06

Requirements:
- numpy
- matplotlib

To run:
$ python nsi_survival_plot.py

"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
g_A = 1.2728
g_S = 1.02
g_T = 0.987
me = 0.511  # positron mass in MeV
delta = 1.29  # neutron-proton mass difference in MeV
L_far = 1500  # far detector distance in meters
L_near = 500  # near detector distance in meters
Delta_m2_31 = 2.52e-3  # eV^2 from arXiv:1811.05487
theta13 = np.arcsin(np.sqrt(0.0841)) / 2  # convert sin^2(2θ13) to θ13 in radians

# Energy range [MeV]
E_nu = np.linspace(1.8, 8, 1000)

# NSI Parameters
Re_S = 0.0
Im_S = 0.0
Re_T = 0.0
Im_T = 0.0

# f_T approximation using Gaussian reactor spectrum
def f_T(E, mean=1.7, sigma=2.5):
    delta_vals = np.linspace(E + me, 20, 1000)
    W = np.exp(-0.5 * ((delta_vals - mean) / sigma) ** 2)
    valid = (delta_vals - E - me) > 0  # ensure sqrt is real
    integrand = (delta_vals - E) * np.sqrt((delta_vals - E - me)*(delta_vals - E + me)) * W
    weight = np.sqrt((delta_vals - E - me)*(delta_vals - E + me)) * W
    num = np.trapz(integrand[valid], delta_vals[valid])
    denom = np.trapz(weight[valid], delta_vals[valid])
    return num / denom if denom > 0 else 0

f_T_vals = np.array([f_T(E) for E in E_nu])

# Compute α and β coefficients
alpha_D = (g_S / (3 * g_A**2 + 1)) * Re_S - (3 * g_A * g_T / (3 * g_A**2 + 1)) * Re_T
alpha_P = (g_T / g_A) * Re_T
alpha_total = alpha_D + alpha_P

beta_D = (g_S / (3 * g_A**2 + 1)) * Im_S - (3 * g_A * g_T / (3 * g_A**2 + 1)) * Im_T
beta_P = (g_T / g_A) * Im_T
beta_total = beta_D + beta_P

# Oscillation probability with NSI
def survival_prob(L, E, alpha, beta, fT):
    arg1 = Delta_m2_31 * L / (4 * E)
    arg2 = Delta_m2_31 * L / (2 * E)
    term1 = np.sin(2 * theta13 + alpha * me / (E - delta) - alpha_P * me * fT) ** 2
    term2 = np.sin(arg2) * np.sin(2 * theta13) * (beta * me / (E - delta) - beta_P * me * fT)
    return 1 - np.sin(arg1) ** 2 * term1 + term2

# Compute probabilities and ratio
P_far = survival_prob(L_far, E_nu, alpha_total, beta_total, f_T_vals)
P_near = survival_prob(L_near, E_nu, alpha_total, beta_total, f_T_vals)
R_far_near = P_far / P_near

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(E_nu, R_far_near, label='Scalar + Tensor NSI')
plt.axhline(1, color='orange', linestyle='--', label='Standard Model (SM)')
plt.xlabel("Neutrino Energy $E_\\nu$ [MeV]")
plt.ylabel("Survival Probability Ratio (Far/Near)")
plt.title("Effect of NSI on Reactor Neutrino Oscillations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
