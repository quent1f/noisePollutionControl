import _env
import numpy

domain = []

# en plus de masquer les noeuds non-intérieurs, on masque les noeuds intérieurs "dans la fractale"
coordinates_to_mask = numpy.argwhere(domain != _env.NODE_INTERIOR)
u = numpy.array(u, copy=True)
mask = numpy.zeros(u.shape, dtype=bool)
mask[coordinates_to_mask[:,0], coordinates_to_mask[:,1]] = True

# let's find the highest(minimum index) row where a node is _env.NODE_ROBIN
coordinates_robin = numpy.argwhere(domain == _env.NODE_ROBIN)
robins_rows = coordinates_robin[:,0]
min_robin_row = numpy.min(robins_rows)
mask[min_robin_row:, :] = True # all rows min_robin_row are masked

u_masked = numpy.ma.array(data=u, mask=mask)