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
num_eigenvalues = 7
gridspacing = 1. / gridsize
x_axis = np.arange(0, 1, gridspacing)

iter_x = np.arange(0, max_iters + 1)

#Exact values
#########################################################
eigenvalues_exact = [i**2 for i in xrange(1, num_eigenvalues + 1)]
eigenvectors_exact = np.zeros((gridsize, num_eigenvalues))
for i in xrange(gridsize):
    for j in xrange(num_eigenvalues):
        eigenvectors_exact[i, j] = np.sin((j+1) * np.pi * x_axis[i])
for j in xrange(num_eigenvalues):
    eigenvectors_exact[:, j] = eigenvectors_exact[:, j] / np.linalg.norm(eigenvectors_exact[:, j])
###########################################################


# Multigrid method
########################################################
# Generate guesses for eigenvalues to use as shift
# 2**4 needed to work
bad_gridsize = 2**4
bad_gridspacing = 1. / bad_gridsize
bad_x_axis = np.arange(0, 1, bad_gridspacing)
bad_laplacian = stencil_maker.laplacian(bad_gridsize)
bad_hamiltonian = (-1 / np.pi ** 2) * bad_laplacian

eigenvalues_shift = np.zeros((1, num_eigenvalues))
eigenvectors_shift = np.zeros((gridsize, num_eigenvalues))


bad_eigenvalues, bad_eigenvectors = sparsela.eigsh(bad_hamiltonian, k=num_eigenvalues, which='SM', tol=tolerance)
print "Initial guess eigenvalues: ", bad_eigenvalues

bad_eigenvectors = np.array(bad_eigenvectors)
w0 = np.zeros((gridsize, 1))

errors = np.zeros((max_iters + 1, num_eigenvalues))
interpolation_matrix = stencil_maker.interpolation(bad_gridsize, gridsize)
for i in xrange(num_eigenvalues):

    eigenvectors_shift[:, i] = interpolation_matrix * bad_eigenvectors[:, i]
    eigenvectors_shift[:, i] /= np.linalg.norm(eigenvectors_shift[:, i])

    error = np.linalg.norm(eigenvectors_exact[:, i].conj().T * eigenvectors_exact[:,  i] - eigenvectors_shift[:, i].conj().T * eigenvectors_shift[:, i])
    errors[0, i] = error

    iters = 0
    while iters < max_iters and error > tolerance:
        w = w0
        w = solver.vcycle(w, eigenvectors_shift[:, i], hamiltonian, stencil_maker, shift=bad_eigenvalues[i], lowest_level=2**4)
        eigenvectors_shift[:, i] = w / np.linalg.norm(w)
        error = np.linalg.norm(eigenvectors_exact[:, i].conj().T * eigenvectors_exact[:, i] - eigenvectors_shift[:, i].conj().T * eigenvectors_shift[:, i])
        print iters, error
        # Ugly because i can't use np.transpose(v) * hamiltonian * v. Matrix multiplication dies
        iters += 1
        errors[iters, i] = error

    eigenvalues_shift[0, i] = np.dot(eigenvectors_shift[:, i].conj().T, hamiltonian.dot(eigenvectors_shift[:, i]))


#######################################################################################

print "Eigenvalues Exact: ", eigenvalues_exact
print "Eigenvalues Shift: ", eigenvalues_shift


fig_error, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    axarray[row_index, column_index].plot(iter_x, errors[:, i])
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("2-norm error na V-cycles")
fig_error.text(0.5, 0.04, 'x', ha='center', va='center')




fig_guess, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    axarray[row_index, column_index].plot(bad_x_axis, bad_eigenvectors[:, i], label="Begingok")
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Begingok op grof grid n = " + str(bad_gridsize))
fig_guess.text(0.5, 0.04, 'x', ha='center', va='center')


fig_exact, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    psi_sq = np.ma.conjugate(eigenvectors_exact[:, i]) * eigenvectors_exact[:, i]
    axarray[row_index, column_index].plot(x_axis, psi_sq)
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Exact")
fig_exact.text(0.5, 0.04, 'x', ha='center', va='center')
fig_exact.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')




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
    psi_sq = np.ma.conjugate(eigenvectors_shift[:, i]) * eigenvectors_shift[:, i]
    axarray[row_index, column_index].plot(x_axis, psi_sq)
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Shift Method")
fig_shift.text(0.5, 0.04, 'x', ha='center', va='center')
fig_shift.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


plt.show()
