from MGCMTSolver import MGCMTSolver
from MGCMTStencilMaker import MGCMTStencilMaker
import numpy as np
import matplotlib.pyplot as plt


stencil_maker = MGCMTStencilMaker()
solver = MGCMTSolver()

n = 2**2
num_vectors = 3

laplacian = stencil_maker.laplacian(n)

f_matrix = np.zeros((n, num_vectors))
for i in xrange(num_vectors):
    f_matrix[:, i] = np.ones((n, 1))[0] * i

######## STANDARD VCYCLE ##########################################################################################
print "STANDARD VCYCLE BEGIN"
x_matrix_single = np.ones((n, num_vectors)) * 4
for i in xrange(num_vectors):
    x_matrix_single[:, i] = solver.vcycle(x_matrix_single[:, i], f_matrix[:, i], laplacian, stencil_maker)

for j in xrange(num_vectors):
    print np.dot(x_matrix_single[:, j].conj().T, laplacian.dot(x_matrix_single[:, j]))
    #  GIVES [-0.382301639189,   -0.257586075778,     -0.840838733687]
print "STANDARD VCYCLE END"
####################################################################################################################

###### STANDARD TWOGRID ############################################################################################
print "STANDARD TWOGRID BEGIN"
x_matrix_single_twogrid = np.ones((n, num_vectors)) * 4
for i in xrange(num_vectors):
    x_matrix_single_twogrid[:, i] = solver.twogrid(x_matrix_single_twogrid[:, i], f_matrix[:, i], laplacian, stencil_maker)

for j in xrange(num_vectors):
    print np.dot(x_matrix_single_twogrid[:, j].conj().T, laplacian.dot(x_matrix_single_twogrid[:, j]))
    #  GIVES [-0.382301639189,   -0.257586075778,     -0.840838733687]
print "STANDARD TWOGRID END"
###################################################################################################################

######## MATRIX VCYCLE ############################################################################################
print "MATRIX VCYCLE BEGIN"
x_matrix = np.ones((n, num_vectors)) * 4
x_matrix = solver.vcycle_matrix(x_matrix, f_matrix, laplacian, stencil_maker)

for j in xrange(num_vectors):
    print np.dot(x_matrix[:, j].conj().T, laplacian.dot(x_matrix[:, j]))
    #  GIVES [-0.382301639189,   -0.257586075778,     -0.840838733687]
print "MATRIX VCYCLE END"
