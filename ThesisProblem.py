from baseCompounds import *
from PotentialWell import PotentialWell
from PotWellSolver import *
import numpy as np
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
from MGCMTProcessor import MGCMTProcessor
import scipy.sparse.linalg as sparsela
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from operator import itemgetter


max_iters = 10  # Maximaal aantal V-cycles
iter_x = np.arange(0, max_iters + 1)

solver = MGCMTSolver()
stencil_maker = MGCMTStencilMaker()
processor = MGCMTProcessor()

gridsize = 2**8
machine_eps = np.finfo(float).eps
tolerance = machine_eps
num_eigenvalues = 6


k = 0  # Zoek de oplossingen in het gamma punt
potwell = PotentialWell("z")  # Opsluiting in de z-richting
GaAs = Compound(GaAsValues)  # Oplossen voor GaAs. Si is ook mogelijk.


potwellsolver_eigs = PotWellSolver(GaAs, potwell, 4)
potwellsolver_eigs.setGridPoints(gridsize)
potwellsolver_eigs.setXRange(-1, 1)
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



# Multigrid method
########################################################
print "Iterative method"
# Generate guesses for fine grid
#2**5 is minimum voor correcte resultaten. 2**4 geeft foute convergentie
bad_gridsize = 2**5
potwellsolver_eigs.setGridPoints(bad_gridsize)
bad_hamiltonian = potwellsolver_eigs.makeMatrix(k)

eigenvalues_shift = np.zeros((1, num_eigenvalues))
eigenvectors_shift = np.zeros((gridsize * potwellsolver_eigs.matrixDim, num_eigenvalues), dtype=complex)


bad_eigenvalues, bad_eigenvectors = sparsela.eigsh(bad_hamiltonian, k=num_eigenvalues, which='SM', tol=tolerance)
# Sorteren van de eigenwaarden en eigenvectoren
bad_eigenvectors = np.array([np.array(bad_eigenvectors[:, i]) for i in xrange(num_eigenvalues)])
bad_eigenvalues, bad_eigenvectors = zip(*sorted(zip(bad_eigenvalues, bad_eigenvectors), key=itemgetter(0)))
bad_eigenvalues = np.array(bad_eigenvalues)

print "Initial guess eigenvalues: %s" % (bad_eigenvalues * unitE)

w0 = np.zeros((gridsize * potwellsolver_eigs.matrixDim, 1))

interpolated_vectors = np.zeros((gridsize * potwellsolver_eigs.matrixDim, num_eigenvalues), dtype=complex)
interpolation_matrix = stencil_maker.interpolation(bad_gridsize * potwellsolver_eigs.matrixDim, gridsize * potwellsolver_eigs.matrixDim)
for i in xrange(num_eigenvalues):
    interpolated_vectors[:, i] = interpolation_matrix * bad_eigenvectors[i]
    interpolated_vectors[:, i] /= np.linalg.norm(interpolated_vectors[:, i])


residuals = np.zeros((len(iter_x), num_eigenvalues))

for i in xrange(num_eigenvalues):

    eigenvectors_shift[:, i] = interpolation_matrix * bad_eigenvectors[i]
    eigenvectors_shift[:, i] /= np.linalg.norm(eigenvectors_shift[:, i])

    shifted_matrix = hamiltonian_sparse - sparse.eye(gridsize * potwellsolver_eigs.matrixDim) * bad_eigenvalues[i] # Create shifted matrix


    iters = 0
    residuals[0, i] = np.linalg.norm(eigenvectors_shift[:, i] - shifted_matrix * w0)
    while iters < max_iters:
        w = w0
        #Lowest good level is 2**5
        w = solver.vcycle(w, eigenvectors_shift[:, i], hamiltonian_sparse, stencil_maker, shift=bad_eigenvalues[i], lowest_level=2**5, smoother=solver.gseidel)
        eigenvectors_shift[:, i] = w / np.linalg.norm(w)

        iters += 1
        residuals[iters, i] = np.linalg.norm(eigenvectors_shift[:, i] - shifted_matrix * w)
        print iters, residuals[iters, i]

    eigenvalues_shift[0, i] = np.dot(eigenvectors_shift[:, i].conj().T, hamiltonian_sparse.dot(eigenvectors_shift[:, i]))
    eigenvalues_shift[0, i] = eigenvalues_shift[0, i].real * unitE  # Real call because eigenvalues have a very very small imaginary part, compuational artifact ?
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
plt.suptitle("Multigrid method")
fig_shift.text(0.5, 0.04, 'x', ha='center', va='center')
fig_shift.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


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
    HH1_int = interpolated_vectors[0:gridsize, i]
    LH1_int = interpolated_vectors[gridsize:2*gridsize, i]
    LH2_int = interpolated_vectors[2*gridsize:3*gridsize, i]
    HH2_int = interpolated_vectors[3*gridsize:4*gridsize, i]

    HH_sq = np.ma.conjugate(HH1_int) * HH1_int + np.ma.conjugate(HH2_int) * HH2_int
    LH_sq = np.ma.conjugate(LH1_int) * LH1_int + np.ma.conjugate(LH2_int) * LH2_int
    axarray[row_index, column_index].plot(x_axis, HH_sq, c='b', label="Heavy Hole")
    axarray[row_index, column_index].plot(x_axis, LH_sq, c='g', label="Light Hole")
    axarray[row_index, column_index].set_title("n = " + str(i+1))
    axarray[row_index, column_index].legend()
plt.suptitle("Begingok voor Multigrid methode")
fig_guess.text(0.5, 0.04, 'x', ha='center', va='center')
fig_guess.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')

fig_residuals, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    axarray[row_index, column_index].plot(iter_x, residuals[:, i], c='b', label="Heavy Hole")
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("2-norm van residu")
fig_residuals.text(0.5, 0.04, 'x', ha='center', va='center')
fig_residuals.text(0.06, 0.5, 'Residu', ha='center', va='center', rotation='vertical')



plt.show()