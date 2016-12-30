import numpy as np
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
from MGCMTProcessor import MGCMTProcessor
import scipy.sparse.linalg as sparsela
import scipy.sparse as sparse
import matplotlib.pyplot as plt


solver = MGCMTSolver()
stencil_maker = MGCMTStencilMaker()
processor = MGCMTProcessor()

gridsize = 2**8
laplacian = stencil_maker.laplacian(gridsize)
hamiltonian = (-1 / np.pi ** 2) * laplacian

machine_eps = np.finfo(float).eps
tolerance = 10 ** (-4)
max_iters = 10
num_eigenvalues = 6
gridspacing = 1. / gridsize
x_axis = np.arange(0, 1, gridspacing)

iter_x = np.arange(0, max_iters + 1)

exact_eigenvalues = [(x + 1) ** 2 for x in xrange(num_eigenvalues)]

lowest_vcycle_grid = 2**3

# Generate guesses for eigenvalues to use as shift
bad_gridsize = 2**3
bad_gridspacing = 1. / bad_gridsize
bad_x_axis = np.arange(0, 1, bad_gridspacing)
bad_laplacian = stencil_maker.laplacian(bad_gridsize)
bad_hamiltonian = (-1 / np.pi ** 2) * bad_laplacian




bad_eigenvalues, bad_eigenvectors = sparsela.eigsh(bad_hamiltonian, k=num_eigenvalues, which='SM', tol=tolerance)
print "Initial guess eigenvalues: ", bad_eigenvalues

bad_eigenvectors = np.array(bad_eigenvectors)


interpolation_matrix = stencil_maker.interpolation(bad_gridsize, gridsize)


# NO GS
########################################################
w0 = np.zeros((gridsize, 1))
eigenvalues_NoGS = np.zeros((max_iters + 1, num_eigenvalues))
eigenvalues_diff_NoGS = np.zeros((max_iters + 1, num_eigenvalues))
eigenvectors_NoGS = np.zeros((gridsize, num_eigenvalues))

for j in xrange(num_eigenvalues):
    eigenvectors_NoGS[:, j] = interpolation_matrix * bad_eigenvectors[:, j]
    eigenvectors_NoGS[:, j] /= np.linalg.norm(eigenvectors_NoGS[:, j])
    eigenvalues_NoGS[0, j] = np.dot(eigenvectors_NoGS[:, j].conj().T, hamiltonian.dot(eigenvectors_NoGS[:, j]))
    eigenvalues_diff_NoGS[0, j] = eigenvalues_NoGS[0, j] - exact_eigenvalues[j]

iters = 0
while iters < max_iters:
    iters += 1
    for j in xrange(num_eigenvalues):
        w = w0
        w = solver.vcycle(w, eigenvectors_NoGS[:, j], hamiltonian, stencil_maker, shift=bad_eigenvalues[j], lowest_level=lowest_vcycle_grid)
        eigenvectors_NoGS[:, j] = w / np.linalg.norm(w)
        eigenvalues_NoGS[iters, j] = np.dot(eigenvectors_NoGS[:, j].conj().T, hamiltonian.dot(eigenvectors_NoGS[:, j]))
        eigenvalues_diff_NoGS[iters, j] = eigenvalues_NoGS[iters, j] - exact_eigenvalues[j]
#######################################################################################


# GS AFTER V-CYCLE
############################################################
w0 = np.zeros((gridsize, 1))
eigenvalues_GS1 = np.zeros((max_iters + 1, num_eigenvalues))
eigenvectors_GS1 = np.zeros((gridsize, num_eigenvalues))
eigenvalues_diff_GS1 = np.zeros((max_iters + 1, num_eigenvalues))


for j in xrange(num_eigenvalues):
    eigenvectors_GS1[:, j] = interpolation_matrix * bad_eigenvectors[:, j]
    eigenvectors_GS1[:, j] /= np.linalg.norm(eigenvectors_GS1[:, j])
    eigenvalues_GS1[0, j] = np.dot(eigenvectors_GS1[:, j].conj().T, hamiltonian.dot(eigenvectors_GS1[:, j]))
    eigenvalues_diff_GS1[0, j] = eigenvalues_GS1[0, j] - exact_eigenvalues[j]

iters = 0
while iters < max_iters:
    iters += 1
    for j in xrange(num_eigenvalues):
        w = w0
        w = solver.vcycle(w, eigenvectors_GS1[:, j], hamiltonian, stencil_maker, shift=bad_eigenvalues[j], lowest_level=lowest_vcycle_grid)
        eigenvectors_GS1[:, j] = w / np.linalg.norm(w)
        eigenvalues_GS1[iters, j] = np.dot(eigenvectors_GS1[:, j].conj().T, hamiltonian.dot(eigenvectors_GS1[:, j]))
        eigenvalues_diff_GS1[iters, j] = eigenvalues_GS1[iters, j] - exact_eigenvalues[j]
        #print iters, eigenvalues_GS1[iters, j]

    eigenvectors_GS1 = processor.gramschmidt(eigenvectors_GS1)
#######################################################################################

# GS IN V-CYCLE
######################################################################################
w0 = np.zeros((gridsize, num_eigenvalues))
eigenvalues_GS2 = np.zeros((max_iters + 1, num_eigenvalues))
eigenvectors_GS2 = np.zeros((gridsize, num_eigenvalues))
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
    w = solver.vcycle_matrix(w, eigenvectors_GS2, hamiltonian, stencil_maker, shifts=bad_eigenvalues, lowest_level=lowest_vcycle_grid)
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


fig_shift, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    psi_sq = eigenvectors_NoGS[:, i].conj().T * eigenvectors_NoGS[:, i]
    axarray[row_index, column_index].plot(x_axis, psi_sq)
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Shift Method")
fig_shift.text(0.5, 0.04, 'x', ha='center', va='center')
fig_shift.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


fig_GS1, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    psi_sq = eigenvectors_GS1[:, i].conj().T * eigenvectors_GS1[:, i]
    axarray[row_index, column_index].plot(x_axis, psi_sq)
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Shift Method")
fig_GS1.text(0.5, 0.04, 'x', ha='center', va='center')
fig_GS1.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


fig_GS2, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    psi_sq = eigenvectors_GS2[:, i].conj().T * eigenvectors_GS2[:, i]
    axarray[row_index, column_index].plot(x_axis, psi_sq)
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Shift Method")
fig_GS2.text(0.5, 0.04, 'x', ha='center', va='center')
fig_GS2.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


plt.show()