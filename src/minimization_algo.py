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
import computeProjectors
#import solutions


### Constant for gradient descent 

EPSILON0 = 10**(-5)
EPSILON1 = 10**(-3)
EPSILON2 = 10**(-4)

###

def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		# print("Robin")
		return 2
	else:
		return 0


def compute_gradient_descent(chi, grad, domain, mu):
	"""This function makes the gradient descent.
	This function has to be used before the 'Projected' function that will project
	the new element onto the admissible space.
	:param chi: density of absorption define everywhere in the domain
	:param grad: parametric gradient associated to the problem
	:param domain: domain of definition of the equations
	:param mu: step of the descent
	:type chi: numpy.array((M,N), dtype=float64
	:type grad: numpy.array((M,N), dtype=float64)
	:type domain: numpy.array((M,N), dtype=int64)
	:type mu: float
	:return chi:
	:rtype chi: numpy.array((M,N), dtype=float64

	.. warnings also: It is important that the conditions be expressed with an "if",
			not with an "elif", as some points are neighbours to multiple points
			of the Robin frontier.
	"""

	(M, N) = numpy.shape(domain)
	# for i in range(0, M):
	# 	for j in range(0, N):
	# 		if domain_omega[i, j] != _env.NODE_ROBIN:
	# 			chi[i, j] = chi[i, j] - mu * grad[i, j]
	# # for i in range(0, M):
	# 	for j in range(0, N):
	# 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
	# 			chi[i,j] = chi[i,j] - mu*grad[i,j]
	# print(domain,'jesuisla')
	#chi[50,:] = chi[50,:] - mu*grad[50,:]
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			#print(i,j)
			#chi[i,j] = chi[i,j] - mu * grad[i,j]
			a = BelongsInteriorDomain(domain[i + 1, j])
			b = BelongsInteriorDomain(domain[i - 1, j])
			c = BelongsInteriorDomain(domain[i, j + 1])
			d = BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:
				# print(i+1,j, "-----", "i+1,j")
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				# print(i - 1, j, "-----", "i - 1, j")
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				# print(i, j + 1, "-----", "i , j + 1")
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				# print(i, j - 1, "-----", "i , j - 1")
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]
	return chi
# a projection method ([0,1] to {0,1})
def sorting_method(domain, chi, V_obj, S):
    """Projecting chi from gamma -> [0;1] to gamma -> {0;1}

    Args:
        chi (numpy.array): chi with values in [0;1]
        V_obj (float): target volume. a ratio
        S (int): number of nodes on frontier. computed during the optimization procedure

    Returns:
        chi_proj: chi with values in {0;1}
    """

    N_obj = round(S*V_obj) # number of nodes necessary to get closest to target
    chi_proj = chi.copy()

    # we want the list of (x,y) coordinates that sorts chi's values in ascending order
    coordinates_wall = numpy.argwhere(domain == _env.NODE_ROBIN)
    values_per_coordinate = chi[coordinates_wall[:,0], coordinates_wall[:,1]]
    sorting_indices = numpy.argsort(-values_per_coordinate) # indices that would sort the values in ascending order
    sorted_coordinates = coordinates_wall[sorting_indices] # coordinates in this order
    # print(sorted_coordinates)
    
    # we set the N_obj biggest values of chi to 1
    chi_proj[sorted_coordinates[:N_obj, 0], sorted_coordinates[:N_obj, 1]] = 1
    # we set its other values to 0
    chi_proj[sorted_coordinates[N_obj:, 0], sorted_coordinates[N_obj:, 1]] = 0

    return chi_proj


# Attention :  j'ai enlevé omega (la fréquence) dans les paramètres (pour le moment je ne vois pas en quoi cela influe sur notre methode d'optimisation)
def optimization_procedure(domain_omega, spacestep, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj, wavenumber, S):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """

    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 100
    energy = []
    while k < numb_iter and mu > EPSILON0:
        print('---- iteration number = ', k)
        ###### On update les conditions aux bord de Robin (puisque l'on a changé chi)
        alpha_rob = Alpha * chi
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ###### On souhaite résoudre le problème adjoint. On ajoute donc un terme source et on met la condition de dirichlet à 0 
        f_adj = -2 * u.conj()
        f_adj_dir = numpy.zeros((M, N), dtype=numpy.complex128)
        p = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f_adj, f_adj_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ene = energy_omega(domain_omega, u, spacestep)
        energy.append(ene)
        print(f"{k} ème energie", ene)
        grad = numpy.real(Alpha * u * p)
        print(f"{k} ème norme L2 du gradient", energy_omega(domain_omega, grad, spacestep))
        while ene >= energy[k] and mu > EPSILON0:
            new_chi = chi.copy()
            new_chi_grad = compute_gradient_descent(new_chi, grad, domain_omega, mu)
            new_chi_grad_projected = computeProjectors.my_compute_projection(new_chi_grad, domain_omega, V_obj, computeProjectors.rectified_linear)
            alpha_rob = Alpha * new_chi_grad_projected
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            ene = energy_omega(domain_omega, u, spacestep)
            if ene < energy[k]:
                # The step is increased if the energy decreased
                mu = mu * 1.1
                chi = new_chi_grad_projected
            else:
                # The step is decreased if the energy increased
                mu = mu * 0.5

        k += 1

        # Sortie de la boucle si la variation d'énergie est inférieure à un seuil
        if k > 1 and abs(ene - energy[k-1]) < 5*10**(-4):
            print("Fin car X_k+1 trop proche de X_k")
            break


    alpha_rob = Alpha * chi
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi_projected = sorting_method(domain_omega, chi, V_obj, S)
    alpha_rob = Alpha * chi_projected
    u_projected = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    
    return chi, energy, u, grad, chi_projected, u_projected



def energy_omega(domain_omega, u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """
    # every element has the same size : spacestep^2
    coordinates_to_mask = numpy.argwhere(domain_omega != _env.NODE_INTERIOR)
    u = numpy.array(u, copy=True)
    mask = numpy.zeros(u.shape, dtype=bool)
    mask[coordinates_to_mask[:,0], coordinates_to_mask[:,1]] = True
    u_masked = numpy.ma.array(data=u, mask=mask)
    u_line = numpy.reshape(u_masked, -1)
    energy = numpy.sum(numpy.absolute(u_line)**2) * (spacestep**2)
    return energy

def energy_off_the_wall():
	pass

# testing energy computation
domain = numpy.array([[_env.NODE_NEUMANN for _ in range(4)],
		  [_env.NODE_INTERIOR for _ in range(4)],
		  [_env.NODE_ROBIN, _env.NODE_ROBIN, _env.NODE_ROBIN, _env.NODE_ROBIN],
		  [_env.NODE_COMPLEMENTARY for _ in range(4)]])

u_real = numpy.random.rand(4,4)
u_img = numpy.random.rand(4,4)
u = u_real + 1j * u_img
print(u)

print(domain[1])
u_1 = numpy.array(u[1], copy=True)
print(numpy.sum(numpy.absolute(u_1**2)))
print(energy_omega(domain_omega=domain, u=u, spacestep=1))