import numpy
import _env
import time

# loading a domain
domain = numpy.load('basic-domain.npy')

# testing a faster way to evaluate the surface
# teacher's solution
t0 = time.time()
(M, N) = numpy.shape(domain)
S = 0
for i in range(M):
    for j in range(N):
        if domain[i, j] == _env.NODE_ROBIN:
            S = S + 1
t1 = time.time()
print(S, t1-t0)
# my solution
t0 = time.time()
S = numpy.count_nonzero(domain == _env.NODE_ROBIN)
t1 = time.time()
print(S, t1-t0)