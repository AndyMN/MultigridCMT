import numpy as np
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
from MGCMTProcessor import MGCMTProcessor
import scipy.sparse.linalg as sparsela
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time

# NOTE: EIGSH USES ITERATIVE METHOD !! (Implicitly Restarted Lanczos Method) (KRYLOV METHODE)
# NOTE: HOW TO HANDLE THE SHIFT DURING A VCYCLE OR TWOGRID (Works right now but pay attention to lowest grid level. Possible that eigenvector can't be represented on lowest grid.)
# NOTE: DEFINE A CONVERGENCE CRITERIUM (Define a boundary for residu AND max iterations in case of no convergence)

solver = MGCMTSolver()
stencil_maker = MGCMTStencilMaker()
processor = MGCMTProcessor()

gridsize = 2**8
laplacian = stencil_maker.laplacian(gridsize)
hamiltonian = (-1 / np.pi ** 2) * laplacian

machine_eps = np.finfo(float).eps
tolerance = 10 ** (-4)
num_eigenvalues = 6
gridspacing = 1. / gridsize
x_axis = np.arange(0, 1, gridspacing)


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


#Eigs method
############################################
eigenvalues_eigs, v = sparsela.eigsh(hamiltonian, which='SM', k=num_eigenvalues, tol=tolerance)

eigenvectors_eigs = np.zeros((gridsize, num_eigenvalues))
for i, array in enumerate(v):
    for j in xrange(num_eigenvalues):
        eigenvectors_eigs[i, j] = array[j]

residu_eigs = np.zeros((1, num_eigenvalues))
for i in xrange(gridsize):
    for j in xrange(num_eigenvalues):
        residu_eigs[0, j] += (eigenvectors_eigs[i, j] - eigenvectors_exact[i, j]) ** 2
###########################################


# Shift method
########################################################
# Generate guesses for eigenvalues to use as shift
bad_gridsize = gridsize / 2
bad_laplacian = stencil_maker.laplacian(bad_gridsize)
bad_hamiltonian = (-1 / np.pi ** 2) * bad_laplacian

eigenvalues_shift = np.zeros((1, num_eigenvalues))
eigenvectors_shift = np.zeros((gridsize, num_eigenvalues))


bad_eigenvalues, bad_eigenvectors = sparsela.eigsh(bad_hamiltonian, k=num_eigenvalues, which='SM', tol=tolerance)
print "Initial guess eigenvalues: ", bad_eigenvalues

bad_eigenvectors = np.array(bad_eigenvectors)
w0 = np.zeros((gridsize, 1))



for i in xrange(num_eigenvalues):
    v0 = processor.interpolate(bad_eigenvectors[:, i], stencil_maker, gridsize)  # Interpolate eigenvector guess to fine grid
    v0 /= np.linalg.norm(v0)
    shifted_matrix = hamiltonian - sparse.eye(gridsize) * bad_eigenvalues[i]  # Create the shifted matrix A-mu*I

    v = v0

    for j in xrange(4):
        #w = solver.vcycle(w0, v, hamiltonian, stencil_maker, shift=bad_eigenvalues[i])
        w = w0
        for k in xrange(5):
            w = solver.vcycle(w, v, hamiltonian, stencil_maker, shift=bad_eigenvalues[i], lowest_level=16)

            #w = solver.twogrid(w, v, hamiltonian, stencil_maker, shift=bad_eigenvalues[i])
            #print j, np.linalg.norm(hamiltonian.dot(w)-bad_eigenvalues[i]*w - v)
            print j, np.linalg.norm(v - shifted_matrix * w, ord=np.inf)

        #w = solver.wjacobi(w0, v, shifted_matrix)
        v = w / np.linalg.norm(w)

        #eigenvalue = bad_eigenvalues[i] + (w.conj().T.dot(v)) / (v.conj().T.dot(v))  # kind of works, but cant derive it

        # Ugly because i can't use np.transpose(v) * hamiltonian * v. Matrix multiplication dies
        eigenvalue = np.dot(v.conj().T, hamiltonian.dot(v))



    eigenvalues_shift[0, i] = eigenvalue
    eigenvectors_shift[:, i] = v[:, 0]


#######################################################################################

print "Eigenvalues Exact: ", eigenvalues_exact
print "Eigenvalues Eigsh: ", eigenvalues_eigs
print "Eigenvalues Shift: ", eigenvalues_shift



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
    axarray[row_index, column_index].plot(x_axis, psi_sq, c=np.random.rand(3,))
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
    axarray[row_index, column_index].plot(x_axis, psi_sq, c=np.random.rand(3,))
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Shift Method")
fig_shift.text(0.5, 0.04, 'x', ha='center', va='center')
fig_shift.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')



# row and column sharing
fig_eigs, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    psi_sq = np.ma.conjugate(eigenvectors_eigs[:, i]) * eigenvectors_eigs[:, i]
    axarray[row_index, column_index].plot(x_axis, psi_sq, c=np.random.rand(3,))
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Eigsh Method")
fig_eigs.text(0.5, 0.04, 'x', ha='center', va='center')
fig_eigs.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


plt.show()