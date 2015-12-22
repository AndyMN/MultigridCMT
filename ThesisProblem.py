from baseCompounds import *
from PotentialWell import PotentialWell
from PotWellSolver import *
import numpy as np
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
import scipy.sparse.linalg as sparsela
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from operator import itemgetter

import time


solver = MGCMTSolver()
stencil_maker = MGCMTStencilMaker()
gridsize = 2**8
machine_eps = np.finfo(float).eps
tolerance = machine_eps
num_eigenvalues = 6
k = 0
potwell = PotentialWell("z")
GaAs = Compound(GaAsValues)


potwellsolver_eigs = PotWellSolver(GaAs, potwell, 4)
potwellsolver_eigs.setGridPoints(gridsize)
x_axis = potwellsolver_eigs.getXAxisVector()


hamiltonian_dense = potwellsolver_eigs.makeMatrix(k)
potwellsolver_eigs.setDense(0)
hamiltonian_sparse = potwellsolver_eigs.makeMatrix(k)


print "Direct method"
eigenvalues_eigs, eigenvectors_eigs = np.linalg.eigh(hamiltonian_dense)
del hamiltonian_dense
print "Direct method end"

eigenvalues_eigs *= unitE
eigenvalues_eigs = sorted(eigenvalues_eigs)
eigenvectors_eigs = np.array(eigenvectors_eigs)

print "Eigenvalues direct method: ", eigenvalues_eigs



# Shift method
########################################################
print "Iterative method"
# Generate guesses for fine grid
bad_gridsize = 2**7
potwellsolver_eigs.setGridPoints(bad_gridsize)
bad_hamiltonian = potwellsolver_eigs.makeMatrix(k)

eigenvalues_shift = np.zeros((1, num_eigenvalues))
eigenvectors_shift = np.zeros((gridsize * potwellsolver_eigs.matrixDim, num_eigenvalues), dtype=complex)


bad_eigenvalues, bad_eigenvectors = sparsela.eigsh(bad_hamiltonian, k=num_eigenvalues, which='SM', tol=tolerance)
bad_eigenvectors = np.array([np.array(bad_eigenvectors[:, i]) for i in xrange(num_eigenvalues)])
bad_eigenvalues, bad_eigenvectors = zip(*sorted(zip(bad_eigenvalues, bad_eigenvectors), key=itemgetter(0)))
bad_eigenvalues = np.array(bad_eigenvalues)

print "Initial guess eigenvalues: %s" % (bad_eigenvalues * unitE)
w0 = np.zeros((gridsize * potwellsolver_eigs.matrixDim, 1))


for i in xrange(num_eigenvalues):
    v0 = solver.interpolate(bad_eigenvectors[i], stencil_maker, gridsize * potwellsolver_eigs.matrixDim)  # Interpolate guess to fine grid
    v0 /= np.linalg.norm(v0)
    shifted_matrix = hamiltonian_sparse - sparse.eye(gridsize * potwellsolver_eigs.matrixDim) * bad_eigenvalues[i] # Create shifted matrix

    v = v0
    for j in xrange(4):
        #v = solver.vcycle(f, v, hamiltonian, stencil_maker, shift=bad_eigenvalues[i])
        #v = solver.twogrid(f, v, shifted_matrix, stencil_maker)
        w = solver.wjacobi(w0, v, shifted_matrix)
        v = w / np.linalg.norm(w)
        # Ugly because i can't use np.transpose(v) * hamiltonian * v. Matrix multiplication dies
        eigenvalue = np.dot(v.conj().T, hamiltonian_sparse.dot(v))

    eigenvalues_shift[0, i] = eigenvalue.real * unitE  # Real call because eigenvalues have a very very small imaginary part, compuational artifact ?
    eigenvectors_shift[:, i] = v[:, 0]
print "Iterative method end"
print "Eigenvalues shift method: ", eigenvalues_shift
#######################################################################################



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
    HH1 = eigenvectors_eigs[0:gridsize, i]
    LH1 = eigenvectors_eigs[gridsize:2*gridsize, i]
    LH2 = eigenvectors_eigs[2*gridsize:3*gridsize, i]
    HH2 = eigenvectors_eigs[3*gridsize:4*gridsize, i]

    HH_sq = np.ma.conjugate(HH1) * HH1 + np.ma.conjugate(HH2) * HH2
    LH_sq = np.ma.conjugate(LH1) * LH1 + np.ma.conjugate(LH2) * LH2
    axarray[row_index, column_index].plot(x_axis, HH_sq, c='b', label="Heavy Hole")
    axarray[row_index, column_index].plot(x_axis, LH_sq, c='g', label="Light Hole")
    axarray[row_index, column_index].set_title("n = " + str(i+1))
    axarray[row_index, column_index].legend()
plt.suptitle("Eigh Method (DIRECT)")
fig_eigs.text(0.5, 0.04, 'x', ha='center', va='center')
fig_eigs.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


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
    HH1_shift = eigenvectors_shift[0:gridsize, i]
    LH1_shift = eigenvectors_shift[gridsize:2*gridsize, i]
    LH2_shift = eigenvectors_shift[2*gridsize:3*gridsize, i]
    HH2_shift = eigenvectors_shift[3*gridsize:4*gridsize, i]

    HH_sq = np.ma.conjugate(HH1_shift) * HH1_shift + np.ma.conjugate(HH2_shift) * HH2_shift
    LH_sq = np.ma.conjugate(LH1_shift) * LH1_shift + np.ma.conjugate(LH2_shift) * LH2_shift
    axarray[row_index, column_index].plot(x_axis, HH_sq, c='b', label="Heavy Hole")
    axarray[row_index, column_index].plot(x_axis, LH_sq, c='g', label="Light Hole")
    axarray[row_index, column_index].set_title("n = " + str(i+1))
    axarray[row_index, column_index].legend()
plt.suptitle("Inverse Iteration Method")
fig_eigs.text(0.5, 0.04, 'x', ha='center', va='center')
fig_eigs.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')

plt.show()