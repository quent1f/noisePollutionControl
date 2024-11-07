# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os
import random



# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import minimization_algo
import compute_alpha
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

def initialize_random_chi(M, N, x, y, V_obj):
    chi = numpy.zeros((M, N), dtype=numpy.float64)
    val = 1.0
    random_list = random.sample(range(1, 2*N), int(N*V_obj))
    for k in random_list:
        chi[int(y[k]), int(x[k])] = val
    return chi




def launch_simulation(N: int, level: int, spacestep: float, wavenumber: float, Alpha: complex, V_obj: float, mu: float, chi_init: int, energy_method=minimization_algo.energy_omega):        # penser à ajouter un moyen de faire un initial chi différent
    """
    Lance la simulation et renvoie les plots interessants
    initial_chi : chi de départ pour la minimisation
    V_obj : pourcentage du volume occupé par le matériau
    mu : initial learning rate 
    chi_init : entier, si = 0 alors initialisation "classique" avec _set_chi, sinon, random initialisation 
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
    
    if chi_init == 0:
        initial_chi = preprocessing._set_chi(M, N, x, y)
    else:
        initial_chi = initialize_random_chi(M, N, x, y, V_obj)
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
                           Alpha, mu, initial_chi, V_obj, wavenumber, S, energy_method=energy_method)
    
    #### Printing varialbes of interest

    print("Energie de départ avec initial_chi:", energy_method(domain_omega, u_init, spacestep))
    print("Energie finale avec chi dans [0, 1]: ", energy_method(domain_omega, u_final, spacestep))
    print("Energie finale avec chi projeté dans {0,1}:", energy_method(domain_omega, u_final_projected, spacestep))
    
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
    return energy





if __name__ == '__main__':
    # -- set parameters of the geometry
    N = 100  # number of point2 along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2 # level of the fractal
    spacestep = 1.0 / N  # mesh size



    # -- set parameters of the partial differential equation

    frequence = 1000
    omega = 2*numpy.pi*frequence
    c = 343 # m/s
    wavenumber = omega/c                        # fréquence f = 200 environ donc w = 2*pi*f = 1200 et k = w/c avec c = 340m/s

    material = 'Melamine'               # Matériau choisi 

    V_obj = 0.6
    mu = 5
    # Alpha = 2.0 - 8.0 * 1j
    Alpha = compute_alpha.compute_alpha(omega, material)[0]
    print("Alpha après calcul", Alpha, "Fréquence", frequence, "Omega", omega, "Nombre d'onde", wavenumber)
    launch_simulation(N, level, spacestep, wavenumber, Alpha, V_obj, mu, energy_method=minimization_algo.energy_omega)    



    """
    1) Tracer l'énergie APRES optimisation en fonction de la densité de matériau. 
    """
    """
   V_obj_list = [0.02*i for i in range(1,50)]
    liste_des_energies = [launch_simulation(N,level,spacestep,wavenumber,Alpha,V_obj, mu)[-1] for V_obj in V_obj_list]


    matplotlib.pyplot.figure(figsize=(8, 5))
    matplotlib.pyplot.plot(V_obj_list[:len(liste_des_energies)], liste_des_energies, marker='o', color='b', linestyle='-', linewidth=2, markersize=6)

    matplotlib.pyplot.xlabel("Densité de matériau (entre 0 et 1)", fontsize=12)
    matplotlib.pyplot.ylabel("Énergie minimale (après opti)", fontsize=12)
    matplotlib.pyplot.title("Évolution de l'énergie après optimisation en fonction de la densité de matériau", fontsize=14)

    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.7)
    matplotlib.pyplot.xlim(0, 1)

    # Afficher le graphique
    matplotlib.pyplot.savefig("energie_min_selon_densité", dpi = 500)

    """








    