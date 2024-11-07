# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import minimization_algo
# import computeAlpha
#import solutions


def computeSurface(domain_omega):
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    return S

def computeVolume(chi, S):
    return numpy.sum(chi)/S

def launch_simulation(N: int, level: int, spacestep: float, wavenumber: float, Alpha: complex, V_obj: float, mu: float):        # penser à ajouter un moyen de faire un initial chi différent
    """
    Lance la simulation et renvoie les plots interessants
    initial_chi : chi de départ pour la minimisation
    V_obj : pourcentage du volume occupé par le matériau
    mu : initial learning rate 
    """
    ##### initialize pde domain
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    ##### conditions de dirichlet : on envoie une onde plainaire
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0

    #####
    S = computeSurface(domain_omega)
    
    initial_chi = preprocessing._set_chi(M, N, x, y)
    preprocessing.set2zero(initial_chi, domain_omega)

    #### Computing alpha
    #Alpha = compute_alpha.compute_alpha(...)
    #### Initial Solving 
    alpha_rob = Alpha * initial_chi
    u_init = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

    #### Optimization 

    chi_final, energy, u_final, grad, chi_final_projected, u_final_projected = minimization_algo.optimization_procedure(domain_omega, spacestep, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, initial_chi, V_obj, wavenumber, S)
    
    #### Printing varialbes of interest

    print("Energie de départ avec initial_chi:", minimization_algo.compute_objective_function(domain_omega, u_init, spacestep))
    print("Energie finale avec chi dans [0, 1]: ", minimization_algo.compute_objective_function(domain_omega, u_final, spacestep))
    print("Energie finale avec chi projeté dans {0,1}:", minimization_algo.compute_objective_function(domain_omega, u_final_projected, spacestep))
    
    print("Volume de chi dans [0,1]", computeVolume(chi_final, S))
    print("Volume de chi projeté dans {0,1}", computeVolume(chi_final_projected, S))
    print("Tableau des énergies", numpy.array(energy))

    #### Plotting (saved on files)

    postprocessing._plot_uncontroled_solution(u_init, initial_chi)
    postprocessing._plot_controled_solution(u_final, chi_final)
    postprocessing._plot_controled_projected_solution(u_final_projected, chi_final_projected)
    err = u_final - u_init
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print('End.')





if __name__ == '__main__':
    # -- set parameters of the geometry
    N = 100  # number of point2 along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2 # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    wavenumber = 10.0                        # fréquence f = 200 environ donc w = 2*pi*f = 1200 et k = w/c avec c = 340m/s

    V_obj = 0.5
    mu = 5
    Alpha = 2.0 - 8.0 * 1j
    # Alpha = computeAlpha(...)

    # initial_chi = preprocessing._set_chi(M, N, x, y)

    launch_simulation(N, level, spacestep, wavenumber, Alpha, V_obj, mu)    



    