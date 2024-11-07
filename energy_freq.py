import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd
import random as rd 

from demo_control_polycopie2024 import *
import sons_et_spectres
import compute_alpha
import preprocessing
import postprocessing
import processing
import _env

from math import pi

materials = {
        "Wood": {
            "phi": 0.5,
            "gamma_p": 7.0 / 5.0,  # Keeping this constant for all materials
            "sigma": 12500.0,
            "rho_0": 600.0,
            "alpha_h": 1.35,
            "c_0": 360.0,
        },
        "Polyester": {
            "phi": 0.9,
            "gamma_p": 7.0 / 5.0,
            "sigma": 20000.0,
            "rho_0": 40.0,
            "alpha_h": 1.2,
            "c_0": 340.0,
        },
        "Melamine": {
            "phi": 0.95,
            "gamma_p": 7.0 / 5.0,
            "sigma": 13000.0,
            "rho_0": 10.0,
            "alpha_h": 1.3,
            "c_0": 340.0,
        },
        "Wool": {
            "phi": 0.9,
            "gamma_p": 7.0 / 5.0,
            "sigma": 22500.0,
            "rho_0": 100.0,
            "alpha_h": 1.4,
            "c_0": 340.0,
        },
    }

# ---------------------------------------------


N = 50  # number of points along x-axis
M = 2 * N  # number of points along y-axis
level = 0 # level of the fractal
spacestep = 1.0 / N  # mesh size
sound_speed = 340.0



domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
f_dir[:, :] = 0.0 # !!!!!
f_dir[0, 0:N] = 1.0 # !!!!!!
chi = preprocessing._set_chi(M, N, x, y)
# chi[:,:] = 1 , enlever # pour fully absorbant
chi = preprocessing.set2zero(chi, domain_omega)
print(chi[50]) 


def compute_objective_function(wavenumber):

    omega =  sound_speed * wavenumber

    computation = compute_alpha.compute_alpha(omega, material_properties) # mettre # ici pour fully absorbant
    # Alpha = - wavenumber * 1j 
    # enlevez le # ici pour fully absorbant
    Alpha = computation[0] # mettre # ici pour fully absorbant
    alpha_rob = Alpha * chi

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


material_properties = materials["Melamine"]

def fractal_shape(wavenumber) :

    global alpha_rob
    global chi
    
    omega =  sound_speed * wavenumber

    computation = compute_alpha.compute_alpha(omega, material_properties) # mettre # ici pour fully absorbant
    # Alpha = - wavenumber * 1j 
    # enlevez le # ici pour fully absorbant
    Alpha = computation[0] # mettre # ici pour fully absorbant 
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

    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    
    chi0 = chi.copy()
    u0 = u.copy()

    #chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
    #                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                      Alpha, mu, chi, V_obj)
    #chi, energy, u, grad = solutions.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                    Alpha, mu, chi, V_obj, mu1, V_0)
    # --- en of optimization

    # chin = chi.copy()
    # un = u.copy()

    # -- plot chi, u, and energy

    postprocessing.plot_domain(domain_omega)
    postprocessing._plot_uncontroled_solution(u0, chi0)
    # postprocessing._plot_controled_solution(un, chin)
    # err = un - u0
    # postprocessing._plot_error(err)

    


start_freq = 100  # Starting frequency (Hz)
end_freq = 1501   # Ending frequency (Hz)
num_points = 300  # Number of points for a smooth curve

filtered_freq = np.linspace(start_freq, end_freq, num_points)

# Compute energy values for level 0

level = 0
domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
f_dir[:, :] = 0.0 # !!!!!
f_dir[0, 0:N] = 1.0 # !!!!!!
energy_values_level_0 = [np.log10(compute_objective_function((2 * pi * freq) / sound_speed)) for freq in filtered_freq]

# ICI, enlevez les # en bas pour fully absorbant

# level = 1
# domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
# beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
# f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
# f_dir[:, :] = 0.0 # !!!!!
# f_dir[0, 0:N] = 1.0 # !!!!!!
# energy_values_level_1 = [np.log10(compute_objective_function((2 * pi * freq) / sound_speed)) for freq in filtered_freq]

# level = 2
# domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)
# beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
# f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
# f_dir[:, :] = 0.0 # !!!!!
# f_dir[0, 0:N] = 1.0 # !!!!!!
# energy_values_level_2 = [np.log10(compute_objective_function((2 * pi * freq) / sound_speed)) for freq in filtered_freq]



# Plot frequencies vs. energies for both levels
plt.figure(figsize=(12, 6))
plt.plot(filtered_freq, energy_values_level_0, label='Level 0', linewidth=2, marker='o')
# plt.plot(filtered_freq, energy_values_level_1, label='Level 1', linewidth=2, marker='o')
# plt.plot(filtered_freq, energy_values_level_2, label='Level 2', linewidth=2, marker='o')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Energy")
plt.title("Energy in function of Frequency for porous on reflective")   
plt.legend()
plt.grid(True)
plt.show()

# Enlevez le # ici si vous voulez afficher les cartes thermiques du u0 et chi0, mettre un wavenumber qui maximise l'énergie par exemple à la place de " ... "
freq = 420
fractal_shape(2*pi*freq/(sound_speed)) 