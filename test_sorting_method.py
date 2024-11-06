import numpy
import matplotlib.pyplot as plt

import testing_HelmolzSolver as test
import _env
import preprocessing
import postprocessing


M = 5
N = 5
domain_omega = numpy.array([[_env.NODE_INTERIOR for _ in range(5)],
                           [_env.NODE_INTERIOR f