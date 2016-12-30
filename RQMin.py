from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
from MGCMTProcessor import MGCMTProcessor
import scipy.sparse.linalg as sparsela
import scipy.sparse as spsparse
import scipy.linalg as scla
import numpy as np
import time
import matplotlib.pyplot as plt

stencil_maker = MGCMTStencilMaker()
solver = MGCMTSolver()
processor = MGCMTProcessor()

gridsize = 2 ** 6

A = (-1 / np.pi ** 2) * stencil_maker.laplacian(gridsize)
M = spsparse.eye(gridsize)

start = time.clock()
eigvals, eigvecs = sparsela.eigsh(A, M=M, k=2, which="SM")
lanczos_time = time.clock() - start

print eigvals

start = time.clock()
x = np.ones((gridsize, 1))
x[:, 0] = np.random.random(gridsize)

rho = 0

for i in xrange(2):
    x, rho = solver.vcycle_rqmg(x, A, M)

print rho

x2 = np.ones((gridsize, 1))
x2[:, 0] = np.random.random(gridsize)

x_matrix = np.zeros((gridsize, 2))
x_matrix[:, 0] = np.array(x[0])
x_matrix[:, 1] = np.array(x2[0])

rho2 = 0

for i in xrange(10):
    x_matrix[:, 1], rho2 = solver.vcycle_rqmg(x_matrix[:, 1], A, M)

    x_matrix = processor.gramschmidt(x_matrix)

print rho2
"""
for i in xrange(4):
    vector_matrix[:, 0], rho = solver.rqmin(A, vector_matrix[:, 0], M, nu=1)
    vector_matrix[:, 1], rho2 = solver.rqmin(A, vector_matrix[:, 1], M, nu=1)


    vector_matrix = processor.gramschmidt(vector_matrix)
"""

rq_time = time.clock() - start

print rho

print "Lanczos time: ", lanczos_time
print "RQMG time: ", rq_time


plt.figure()
plt.plot(x / np.linalg.norm(x))

plt.figure()
plt.plot(x2 / np.linalg.norm(x2))

plt.show()
