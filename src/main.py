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

CELERITY = 343 # célérité du son 

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
    random_list = random.sample(range(0, len(x)-1), int(N*V_obj))
    for k in random_list:
        chi[int(y[k]), int(x[k])] = val
    return chi



def launch_simulation(N: int, level: int, spacestep: float, wavenumber: float, V_obj: float, mu: float, chi_init=0):        # penser à ajouter un moyen de faire un initial chi différent
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
    omega = CELERITY*wavenumber
    Alpha = compute_alpha.compute_alpha(omega, 'Melamine')[0]
    print("Alpha après calcul", Alpha, "Fréquence", omega/(2*numpy.pi), "Omega", omega, "Nombre d'onde", wavenumber)

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
    return energy





if __name__ == '__main__':
    # -- set parameters of the geometry
    N = 200  # number of point2 along x-axis
    M = 2 * N  # number of points along y-axis
    level = 3 # level of the fractal
    spacestep = 1.0 / N  # mesh size



    # -- set parameters of the partial differential equation

    frequence = 440
    omega = 2*numpy.pi*frequence
    # c = 343 # m/s
    wavenumber = omega/CELERITY                        # fréquence f = 200 environ donc w = 2*pi*f = 1200 et k = w/c avec c = 340m/s

    # material = 'Melamine'               # Matériau choisi 

    V_obj = 0.95
    mu = 5
    # Alpha = 2.0 - 8.0 * 1j
    launch_simulation(N, level, spacestep, wavenumber, V_obj, mu, chi_init=1)    

    

    """
    1) Tracer l'énergie APRES optimisation en fonction de la densité de matériau. 
    """

    """
    V_obj_list = [0.02*i for i in range(1,50)]
    liste_energies_initiales = []
    liste_energies_finales = []
    for V_obj in V_obj_list:
        energies = launch_simulation(N,level,spacestep,wavenumber,V_obj, mu) 
        liste_energies_finales.append(energies[-1])
        liste_energies_initiales.append(energies[0])

    matplotlib.pyplot.figure(figsize=(10, 5))
    matplotlib.pyplot.plot(V_obj_list[:len(liste_energies_finales)], liste_energies_finales, marker='o', color='b', linestyle='-', linewidth=2, markersize=3)
    # matplotlib.pyplot.plot(V_obj_list[:len(liste_energies_finales)], liste_energies_initiales, marker='x', color='r', linestyle='-', linewidth=2, markersize=3)

    matplotlib.pyplot.xlabel("Densité de matériau (entre 0 et 1)", fontsize=12)
    matplotlib.pyplot.ylabel("Énergie minimale (après opti)", fontsize=12)
    matplotlib.pyplot.title("Évolution de l'énergie après optimisation en fonction de la densité de matériau", fontsize=14)

    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.7)
    matplotlib.pyplot.xlim(0, 1)

    # Afficher le graphique
    matplotlib.pyplot.savefig("energie_min_selon_densité", dpi = 500)
    """




    """
    2) Tracer l'énergie POST optimisation en fonction de la fréquence de l'onde 
    """

    """
    freq_list = numpy.linspace(50, 4000, 200)
    omega_list = freq_list*2*numpy.pi
    wavenumber_list = omega_list/CELERITY

    liste_energies_initiales = []
    liste_energies_finales = []
    for wavenumber in wavenumber_list:
        energies = launch_simulation(N,level,spacestep,wavenumber,V_obj, mu) 
        liste_energies_finales.append(energies[-1])
        liste_energies_initiales.append(energies[0])


    matplotlib.pyplot.figure(figsize=(11, 5))
    matplotlib.pyplot.plot(freq_list[:len(liste_energies_finales)], liste_energies_finales, marker='o', color='b', linestyle='-', linewidth=2, markersize=3)
    # matplotlib.pyplot.plot(freq_list[:len(liste_energies_finales)], liste_energies_initiales, marker='x', color='r', linestyle='-', linewidth=2, markersize=3)

    matplotlib.pyplot.xlabel("Fréquence de l'onde planaire en entrée", fontsize=12)
    matplotlib.pyplot.ylabel("Énergie minimale (après opti)", fontsize=12)
    matplotlib.pyplot.title("Évolution de l'énergie après optimisation en fonction de la fréquence. beta = 0.5, N=100, level = 2" , fontsize=14)

    matplotlib.pyplot.grid(True, linestyle='--', alpha=0.7)
    matplotlib.pyplot.xlim(0, 4000)

    # Afficher le graphique
    matplotlib.pyplot.savefig("energie_min_selon_frequence", dpi = 500)

    """

    """
    3) Tentative :
    Je vais essayer de lancer K fois l'algo avec un départ $chi_0$ différent pour éviter de me bloquer dans un minimum local et voir ce que ca donne
    """

    """
    liste_energies_initiales = []
    liste_energies_finales = []
    for _ in range(50):
        energies = launch_simulation(N,level,spacestep,wavenumber,V_obj, mu, chi_init=1) 
        liste_energies_finales.append(energies[-1])
        liste_energies_initiales.append(energies[0])

 
    print("énergies de départ:", liste_energies_initiales)
    print("énergies finales:", liste_energies_finales)
    print("max énergies départ:", max(liste_energies_initiales))
    print("min énergies départ:", min(liste_energies_initiales))
    print("max énergies finale:", max(liste_energies_finales))
    print("min énergies finale:", min(liste_energies_finales))

    print("moyenne énergies départ:", numpy.mean(liste_energies_initiales))
    print("moyenne énergies finales:", numpy.mean(liste_energies_finales))
    print("écart-type énergies départ:", numpy.std(liste_energies_initiales))
    print("écart-type énergies finales:", numpy.std(liste_energies_finales))

    """

    """
    4) Optimize the energy at once for several different frequencies 
    
    To do this we use the code in minimization_algo.py with optimization_procedure_2. It computes the sum of the energies for a given frequency band. 
    """



