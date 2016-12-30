import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
from MGCMTProcessor import MGCMTProcessor
import scipy.sparse.linalg as sparsela
import scipy.sparse as sparse
import matplotlib.pyplot as plt

stencil_maker = MGCMTStencilMaker()
solver = MGCMTSolver()
processor = MGCMTProcessor()

max_iters = 5
iter_x = np.arange(0, max_iters + 1)
num_eigenvalues = 10
machine_eps = np.finfo(float).eps
tolerance = machine_eps
gridsize = 2**7
gridspacing = 1. / gridsize
L = 1
x = np.arange(0, L, gridspacing)
y = x
X, Y = np.meshgrid(x, y)
laplacian = stencil_maker.laplacian(gridsize, dimension="2d")
hamiltonian = (-1 / np.pi ** 2) * laplacian


nx = [1, 1, 2, 2, 3, 1, 2, 3, 1, 4]
ny = [1, 2, 1, 2, 1, 3, 3, 2, 4, 1]
n = zip(nx, ny)
exact_eigenvalues = [couple[0]**2 + couple[1]**2 for couple in n]

lowest_vcycle_grid = 2**3

# Generate guesses for eigenvalues to use as shift
bad_gridsize = 2**4
bad_gridspacing = 1. / bad_gridsize
bad_x = np.arange(0, 1, bad_gridspacing)
bad_y = bad_x
xx, yy = np.meshgrid(bad_x, bad_y)
bad_laplacian_2d = stencil_maker.laplacian(bad_gridsize, dimension="2d")
bad_hamiltonian = -1. / np.pi ** 2 * bad_laplacian_2d


bad_eigenvalues, bad_eigenvectors = sparsela.eigsh(bad_hamiltonian, which="SM", tol=tolerance, k=num_eigenvalues)
print "Initial guess eigenvalues: ", bad_eigenvalues

bad_eigenvectors = np.array(bad_eigenvectors)

interpolation_matrix = stencil_maker.interpolation(bad_gridsize, gridsize, dimension="2d")


# NO GS
##################################################################################################
w0 = np.zeros((gridsize**2, 1))
eigenvalues_NoGS = np.zeros((max_iters + 1, num_eigenvalues))
eigenvalues_diff_NoGS = np.zeros((max_iters + 1, num_eigenvalues))
eigenvectors_NoGS = np.zeros((gridsize**2, num_eigenvalues))

for i in xrange(num_eigenvalues):
    eigenvectors_NoGS[:, i] = interpolation_matrix * bad_eigenvectors[:, i]
    eigenvectors_NoGS[:, i] /= np.linalg.norm(eigenvectors_NoGS[:, i])
    eigenvalues_NoGS[0, i] = np.dot(eigenvectors_NoGS[:, i].conj().T, hamiltonian.dot(eigenvectors_NoGS[:, i]))
    eigenvalues_diff_NoGS[0, i] = eigenvalues_NoGS[0, i] - exact_eigenvalues[i]

iters = 0
while iters < max_iters:
    iters += 1
    for i in xrange(num_eigenvalues):
        w = w0
        w = solver.vcycle(w, eigenvectors_NoGS[:, i], hamiltonian, stencil_maker, shift=bad_eigenvalues[i], dimension="2d", lowest_level=lowest_vcycle_grid)
        eigenvectors_NoGS[:, i] = w / np.linalg.norm(w)
        eigenvalues_NoGS[iters, i] = np.dot(eigenvectors_NoGS[:, i].conj().T, hamiltonian.dot(eigenvectors_NoGS[:, i]))
        eigenvalues_diff_NoGS[iters, i] = eigenvalues_NoGS[iters, i] - exact_eigenvalues[i]
################################################################################################



# GS AFTER VCYCLE
################################################################################################
w0 = np.zeros((gridsize**2, 1))
eigenvalues_GS1 = np.zeros((max_iters + 1, num_eigenvalues))
eigenvalues_diff_GS1 = np.zeros((max_iters + 1, num_eigenvalues))
eigenvectors_GS1 = np.zeros((gridsize**2, num_eigenvalues))

for i in xrange(num_eigenvalues):
    eigenvectors_GS1[:, i] = interpolation_matrix * bad_eigenvectors[:, i]
    eigenvectors_GS1[:, i] /= np.linalg.norm(eigenvectors_GS1[:, i])
    eigenvalues_GS1[0, i] = np.dot(eigenvectors_GS1[:, i].conj().T, hamiltonian.dot(eigenvectors_GS1[:, i]))
    eigenvalues_diff_GS1[0, i] = eigenvalues_GS1[0, i] - exact_eigenvalues[i]

iters = 0
while iters < max_iters:
    iters += 1
    for i in xrange(num_eigenvalues):
        w = w0
        w = solver.vcycle(w, eigenvectors_GS1[:, i], hamiltonian, stencil_maker, shift=bad_eigenvalues[i], dimension="2d", lowest_level=lowest_vcycle_grid)
        eigenvectors_GS1[:, i] = w / np.linalg.norm(w)
        eigenvalues_GS1[iters, i] = np.dot(eigenvectors_GS1[:, i].conj().T, hamiltonian.dot(eigenvectors_GS1[:, i]))
        eigenvalues_diff_GS1[iters, i] = eigenvalues_GS1[iters, i] - exact_eigenvalues[i]

    eigenvectors_GS1 = processor.gramschmidt(eigenvectors_GS1)
######################################################################################################

# GS IN V-CYCLE
######################################################################################
w0 = np.zeros((gridsize**2, num_eigenvalues))
eigenvalues_GS2 = np.zeros((max_iters + 1, num_eigenvalues))
eigenvectors_GS2 = np.zeros((gridsize**2, num_eigenvalues))
eigenvalues_diff_GS2 = np.zeros((max_iters + 1, num_eigenvalues))


for j in xrange(num_eigenvalues):
    eigenvectors_GS2[:, j] = interpolation_matrix * bad_eigenvectors[:, j]
    eigenvectors_GS2[:, j] /= np.linalg.norm(eigenvectors_GS2[:, j])
    eigenvalues_GS2[0, j] = np.dot(eigenvectors_GS2[:, j].conj().T, hamiltonian.dot(eigenvectors_GS2[:, j]))
    eigenvalues_diff_GS2[0, j] = eigenvalues_GS2[0, j] - exact_eigenvalues[j]

iters = 0
while iters < max_iters:
    iters += 1
    w = w0
    w = solver.vcycle_matrix(w, eigenvectors_GS2, hamiltonian, stencil_maker, shifts=bad_eigenvalues, dimension="2d", lowest_level=lowest_vcycle_grid)
    for j in xrange(num_eigenvalues):
        eigenvectors_GS2[:, j] = w[:, j] / np.linalg.norm(w[:, j])
        eigenvalues_GS2[iters, j] = np.dot(eigenvectors_GS2[:, j].conj().T, hamiltonian.dot(eigenvectors_GS2[:, j]))
        eigenvalues_diff_GS2[iters, j] = eigenvalues_GS2[iters, j] - exact_eigenvalues[j]
#######################################################################################


print "Exact eigenvalues: ", exact_eigenvalues
print "Eigenvalues NoGS: ", eigenvalues_NoGS[-1, :]
print "Eigenvalues GS1: ", eigenvalues_GS1[-1, :]
print "Eigenvalues GS2: ", eigenvalues_GS2[-1, :]



fig_eigenvalues_NoGS, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
row_index = 0
column_index = 0
for i in xrange(num_eigenvalues):
    if i != 0:
        if i % 2 == 0:
            row_index += 1
            if column_index > 0:
                column_index -= 1
        else:
            column_index += 1
    axarray[row_index, column_index].plot(iter_x[1:], eigenvalues_diff_NoGS[1:, i])
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Eigenwaarde na V-cycles zonder Gram-Schmidt")
fig_eigenvalues_NoGS.text(0.5, 0.04, 'x', ha='center', va='center')


fig_eigenvalues_GS1, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
row_index = 0
column_index = 0
for i in xrange(num_eigenvalues):
    if i != 0:
        if i % 2 == 0:
            row_index += 1
            if column_index > 0:
                column_index -= 1
        else:
            column_index += 1
    axarray[row_index, column_index].plot(iter_x[1:], eigenvalues_diff_GS1[1:, i])
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Eigenwaarde na V-cycles met Gram-Schmidt erna")
fig_eigenvalues_GS1.text(0.5, 0.04, 'x', ha='center', va='center')


fig_eigenvalues_GS2, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
row_index = 0
column_index = 0
for i in xrange(num_eigenvalues):
    if i != 0:
        if i % 2 == 0:
            row_index += 1
            if column_index > 0:
                column_index -= 1
        else:
            column_index += 1
    axarray[row_index, column_index].plot(iter_x[1:], eigenvalues_diff_GS2[1:, i])
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Eigenwaarde na V-cycles met Gram-Schmidt erin")
fig_eigenvalues_GS2.text(0.5, 0.04, 'x', ha='center', va='center')