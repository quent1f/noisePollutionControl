import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd
import random as rd 

import demo_control_polycopie2024
import sons_et_spectres
import compute_alpha
import preprocessing
import postprocessing
import processing
import _env

from math import pi

file = pd.read_csv("ei-pollution-interieure-groupe-3\sons_et_spectres\spectre_bureau_2048.txt", delimiter="\t")
file["Fréquence (Hz)"] = file["Fréquence (Hz)"].str.replace(",", ".", regex=False)
file["Fréquence (Hz)"] = pd.to_numeric(file["Fréquence (Hz)"], errors='coerce')
filtered_freq = file[(file["Fréquence (Hz)"] >= 100) & (file["Fréquence (Hz)"] <= 1000)]["Fréquence (Hz)"]


print(filtered_freq)


# ---------------------------------------------


N = 50  # number of points along x-axis
M = 2 * N  # number of points along y-axis
level = 0 # level of the fractal
spacestep = 1.0 / N  # mesh size
sound_speed = 340.0

domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
print(beta_rob)
f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
f_dir[:, :] = 0.0
f_dir[0, 0:N] = 1.0
chi = preprocessing._set_chi(M, N, x, y)
chi = preprocessing.set2zero(chi, domain_omega)


def compute_objective_function(wavenumber):

    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    # every element has the same size : spacestep^2
    coordinates_to_mask = np.argwhere(domain_omega != _env.NODE_INTERIOR)
    u = np.array(u, copy=True)
    mask = np.zeros(u.shape, dtype=bool)
    mask[coordinates_to_mask[:,0], coordinates_to_mask[:,1]] = True
    u_masked = np.ma.array(data=u, mask=mask)
    u_line = np.reshape(u_masked, -1)
    energy = np.sum(np.absolute(u_line)**2) * (spacestep**2)

    return energy



def fractal_shape(wavenumber) :

    global alpha_rob
    alpha_rob[:, :] = - wavenumber * 1j
    omega =  sound_speed * wavenumber
    # -- define absorbing material
    Alpha = compute_alpha(omega) # En attendant la fonction compute_alpha
    # -- this is the function you have written during your project
    #import compute_alpha
    #Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = np.sum(np.sum(chi)) / S  # constraint on the density
    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution

    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = compute_objective_function(wavenumber)
    # chi, energy, u, grad = your_optimization_procedure(...)
    #chi, energy, u, grad = solutions.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                    Alpha, mu, chi, V_obj, mu1, V_0)
    # --- en of optimization

    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy

    postprocessing._plot_uncontroled_solution(u0, chi0)

    

energy_values = []

# Loop over each frequency and calculate the energy
for freq in filtered_freq:
    wavenumber = (2 * pi * freq) / sound_speed
    # Calculate and store the energy for each frequency
    energy = compute_objective_function(wavenumber)
    energy_values.append(energy)  # Store energy for plotting later

# Plot frequencies vs. energies after the loop
plt.figure(figsize=(12, 6))
plt.plot(filtered_freq, energy_values, marker='o')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Energy")
plt.title("Energy vs Frequency")
plt.show()

freq = rd.choice(filtered_freq)
print(freq)
wavenumber = (2 * pi * freq)/(sound_speed)
fractal_shape(wavenumber)

