import _env
import numpy
import preprocessing

def rectified_linear(vect):
    return numpy.maximum(0, numpy.minimum(vect, 1))

def sigmoid(vect, a):
    return 1 / (1+numpy.exp(-a*(vect-0.5)))

#TODO
def flattened_linear(vect, delta):
    pass

def my_compute_projection(chi, domain, V_obj, phi=rectified_linear):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint. a ratio
    :type chi: numpy.array((M,N), dtype=float64)
    :type domain: numpy.array((M,N), dtype=complex128)
    :type float: float
    :rtype:
    """
    S = numpy.count_nonzero(domain == _env.NODE_ROBIN) # will be used at each iteration

    chi = preprocessing.set2zero(chi, domain) # making sure that chi is 0 on the rest of the mesh
    V = numpy.sum(chi) / S

    # dichotomy boundaries
    debut = -numpy.max(chi)
    fin = numpy.max(chi)

    chi_copy = chi.copy()
    l = 0
    ecart = fin - debut
    while ecart > 10 ** -4: # we must stop looking for l when close enough
        # calcul du milieu
        l = (debut + fin) / 2
        chi = phi(chi_copy+l)
        chi = preprocessing.set2zero(chi, domain) # making sure that chi is 0 on the rest of the mesh
        V = numpy.sum(chi) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi

# We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space