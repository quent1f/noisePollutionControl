import numpy
import matplotlib.pyplot as plt

import testing_HelmolzSolver as test
import _env
import preprocessing
import postprocessing


M = 5
N = 5
domain_omega = numpy.array([[_env.NODE_INTERIOR for _ in range(5)],
                           [_env.NODE_INTERIOR for _ in range(5)],
                           [_env.NODE_INTERIOR, _env.NODE_ROBIN, _env.NODE_INTERIOR, _env.NODE_INTERIOR, _env.NODE_INTERIOR],
                           [_env.NODE_ROBIN, _env.NODE_INTERIOR, _env.NODE_ROBIN, _env.NODE_ROBIN, _env.NODE_ROBIN],
                           [_env.NODE_INTERIOR for _ in range(5)]])
print(domain_omega)
chi_result = numpy.array([[0 for _ in range(5)],
                           [0 for _ in range(5)],
                           [0, 0.99, 0, 0, 0],
                           [0.3104, 0, 0.99, 0.28, 0.45], # au hasard, somme proche de S*V_obj
                           [_env.NODE_INTERIOR for _ in range(5)]])
print(chi_result)
S = 0  # surface of the fractal
for i in range(0, M):
    for j in range(0, N):
        if domain_omega[i, j] == _env.NODE_ROBIN:
            S += 1
print(S)
V_obj = 3/5
print(S*V_obj)
print(test.sorting_method(chi=chi_result, V_obj=V_obj, S=S, domain=domain_omega))