import numpy as np
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
import scipy.sparse.linalg as sparsela
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time

## GIVES THE CORRECT INTERPOLATION MATRICES NOW (PREFACTOR FOR RESTRICTION WAS WRONGLY CALCULATED)


solver = MGCMTSolver()
stencil_maker = MGCMTStencilMaker()
gridsize = 2**4

# Restrict from n to n/2
restriction_operator = stencil_maker.restriction(gridsize, gridsize/2)

print type(restriction_operator)
print np.shape(restriction_operator)
print restriction_operator.toarray()

# Interpolate from n/2 to n
interpolation_operator = stencil_maker.interpolation(gridsize/2, gridsize)

print type(interpolation_operator)
print np.shape(interpolation_operator)
print interpolation_operator.toarray()


A = stencil_maker.laplacian(gridsize)

coarse_operator = restriction_operator * A * interpolation_operator

print type(coarse_operator)
print np.shape(coarse_operator)
print coarse_operator.toarray()