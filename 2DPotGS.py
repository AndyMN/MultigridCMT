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
num_eigenvals = 10
machine_eps = np.finfo(float).eps
tolerance = machine_eps
gridsize = 2**6
gridspacing = 1. / gridsize
L = 1
x = np.arange(0, L, gridspacing)
y = x
X, Y = np.meshgrid(x, y)
laplacian = stencil_maker.laplacian(gridsize, dimension="2d")
hamiltonian = (-1 / np.pi ** 2) * laplacian



### EXACT VALUES ###################################
####################################################
nx = [1, 1, 2, 2, 3, 1, 2, 3, 1, 4]
ny = [1, 2, 1, 2, 1, 3, 3, 2, 4, 1]
n = zip(nx, ny)
exact_eigenvals = [couple[0]**2 + couple[1]**2 for couple in n]

exact_eigenvectors = np.zeros((gridsize**2, num_eigenvals))
for k, couple in enumerate(n):

    Psi = np.zeros((gridsize, gridsize))

    for i, xval in enumerate(x):
        for j, yval in enumerate(y):
            Psi[i, j] = 2 * np.sin(couple[0] * np.pi * xval) * np.sin(couple[1] * np.pi * yval)

    Psi /= np.linalg.norm(Psi)
    Psi = Psi.reshape(gridsize**2, 1)
    exact_eigenvectors[:, k] = Psi[:, 0]
#########################################################

# Multigrid method
########################################################
bad_gridsize = 2**4
bad_gridspacing = 1. / bad_gridsize
bad_x = np.arange(0, 1, bad_gridspacing)
bad_y = bad_x
xx, yy = np.meshgrid(bad_x, bad_y)
bad_laplacian_2d = stencil_maker.laplacian(bad_gridsize, dimension="2d")
bad_hamiltonian = -1. / np.pi ** 2 * bad_laplacian_2d


bad_eigenvals, bad_eigenvectors = sparsela.eigsh(bad_hamiltonian, which="SM", tol=tolerance, k=num_eigenvals)
print "Initial guess eigenvalues: ", bad_eigenvals

bad_eigenvectors = np.array(bad_eigenvectors)

w0 = np.zeros((gridsize**2, 1))
errors = np.zeros((max_iters + 1, num_eigenvals))
eigenvalues = np.zeros((max_iters + 1, num_eigenvals))

eigenvalues_MG = np.zeros((1, num_eigenvals))
eigenvectors_MG = np.zeros((gridsize**2, num_eigenvals))




for i in xrange(num_eigenvals):
    interpolation_matrix = stencil_maker.interpolation(bad_gridsize, gridsize, dimension="2d")
    eigenvectors_MG[:, i] = interpolation_matrix * bad_eigenvectors[:, i]
    eigenvectors_MG[:, i] /= np.linalg.norm(eigenvectors_MG[:, i])

    shifted_matrix = hamiltonian - sparse.eye(gridsize**2) * bad_eigenvals[i]  # Create the shifted matrix A-mu*I
    eigenvalues[0, i] = np.dot(eigenvectors_MG[:, i].conj().T, hamiltonian.dot(eigenvectors_MG[:, i]))

    #error = np.linalg.norm(exact_eigenvectors[:, i].conj().T * exact_eigenvectors[:, i] - eigenvectors_MG[:, i].conj().T * eigenvectors_MG[:, i])
    error = np.linalg.norm(shifted_matrix.dot(eigenvectors_MG[:, i]))
    errors[0, i] = error

iters = 0
while iters < max_iters:
    iters += 1
    for i in xrange(num_eigenvals):
        w = w0
        w = solver.vcycle(w, eigenvectors_MG[:, i], hamiltonian, stencil_maker, shift=bad_eigenvals[i], dimension="2d", lowest_level=2**3)
        eigenvectors_MG[:, i] = w / np.linalg.norm(w)
        #errors[iters, i] = np.linalg.norm(exact_eigenvectors[:, i].conj().T * exact_eigenvectors[:, i] - eigenvectors_MG[:, i].conj().T * eigenvectors_MG[:, i])
        shifted_matrix = hamiltonian - sparse.eye(gridsize**2) * bad_eigenvals[i]  # Create the shifted matrix A-mu*I
        errors[iters, i] = np.linalg.norm(shifted_matrix.dot(eigenvectors_MG[:, i]))

        print iters, errors[iters, i]

        eigenvalues[iters, i] = np.dot(eigenvectors_MG[:, i].conj().T, hamiltonian.dot(eigenvectors_MG[:, i]))

    eigenvectors_MG = processor.gramschmidt(eigenvectors_MG)

for i in xrange(num_eigenvals):
    eigenvalues_MG[0, i] = np.dot(eigenvectors_MG[:, i].conj().T, hamiltonian.dot(eigenvectors_MG[:, i]))

#######################################################################################
############################### EIGSH LANCZOS ########################################

eigenvalues_eigsh, eigenvectors_eigsh = sparsela.eigsh(hamiltonian, k=num_eigenvals, which="SM", tol=tolerance)



print "Exact eigenvalues:", exact_eigenvals
print "Multigrid eigenvalues:", eigenvalues_MG
print "Lanczos eigenvalues:", eigenvalues_eigsh




fig_state1 = plt.figure(1)
fig_state1.suptitle("State (1, 1)")
state_MG = eigenvectors_MG[:, 0].reshape(gridsize, gridsize)
state_exact = exact_eigenvectors[:, 0].reshape(gridsize, gridsize)
state_eigsh = eigenvectors_eigsh[:, 0].reshape(gridsize, gridsize)

#ax_states = fig_state1.add_subplot(1, 3, 1, projection='3d')
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_MG) * state_MG)
#ax_states.set_title("Multigrid")
#ax_states = fig_state1.add_subplot(1, 3, 2, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_eigsh) * state_eigsh)
#ax_states.set_title("Lanczos")
#ax_states = fig_state1.add_subplot(1, 3, 3, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_exact) * state_exact)
#ax_states.set_title("Exact")

ax_states = fig_state1.add_subplot(1, 3, 1)
ax_states.imshow(np.ma.conjugate(state_MG) * state_MG)
ax_states.set_title("Multigrid")
ax_states = fig_state1.add_subplot(1, 3, 2)
ax_states.imshow(np.ma.conjugate(state_eigsh) * state_eigsh)
ax_states.set_title("Lanczos")
ax_states = fig_state1.add_subplot(1, 3, 3)
ax_states.imshow(np.ma.conjugate(state_exact) * state_exact)
ax_states.set_title("Exact")



fig_state2 = plt.figure(2)
fig_state2.suptitle("State (1, 2)")
state_MG = eigenvectors_MG[:, 1].reshape(gridsize, gridsize)
state_exact = exact_eigenvectors[:, 1].reshape(gridsize, gridsize)
state_eigsh = eigenvectors_eigsh[:, 1].reshape(gridsize, gridsize)

#ax_states = fig_state2.add_subplot(1, 3, 1, projection='3d')
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_MG) * state_MG)
#ax_states.set_title("Multigrid")
#ax_states = fig_state2.add_subplot(1, 3, 2, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_eigsh) * state_eigsh)
#ax_states.set_title("Lanczos")
#ax_states = fig_state2.add_subplot(1, 3, 3, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_exact) * state_exact)
#ax_states.set_title("Exact")

ax_states = fig_state2.add_subplot(1, 3, 1)
ax_states.imshow(np.ma.conjugate(state_MG) * state_MG)
ax_states.set_title("Multigrid")
ax_states = fig_state2.add_subplot(1, 3, 2)
ax_states.imshow(np.ma.conjugate(state_eigsh) * state_eigsh)
ax_states.set_title("Lanczos")
ax_states = fig_state2.add_subplot(1, 3, 3)
ax_states.imshow(np.ma.conjugate(state_exact) * state_exact)
ax_states.set_title("Exact")


fig_state3 = plt.figure(3)
fig_state3.suptitle("State (2, 1)")
state_MG = eigenvectors_MG[:, 2].reshape(gridsize, gridsize)
state_exact = exact_eigenvectors[:, 2].reshape(gridsize, gridsize)
state_eigsh = eigenvectors_eigsh[:, 2].reshape(gridsize, gridsize)
#ax_states = fig_state3.add_subplot(1, 3, 1, projection='3d')
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_MG) * state_MG)
#ax_states.set_title("Multigrid")
#ax_states = fig_state3.add_subplot(1, 3, 2, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_eigsh) * state_eigsh)
#ax_states.set_title("Lanczos")
#ax_states = fig_state3.add_subplot(1, 3, 3, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_exact) * state_exact)
#ax_states.set_title("Exact")

ax_states = fig_state3.add_subplot(1, 3, 1)
ax_states.imshow(np.ma.conjugate(state_MG) * state_MG)
ax_states.set_title("Multigrid")
ax_states = fig_state3.add_subplot(1, 3, 2)
ax_states.imshow(np.ma.conjugate(state_eigsh) * state_eigsh)
ax_states.set_title("Lanczos")
ax_states = fig_state3.add_subplot(1, 3, 3)
ax_states.imshow(np.ma.conjugate(state_exact) * state_exact)
ax_states.set_title("Exact")

fig_state4 = plt.figure(4)
fig_state4.suptitle("State (2, 2)")
state_MG = eigenvectors_MG[:, 3].reshape(gridsize, gridsize)
state_exact = exact_eigenvectors[:, 3].reshape(gridsize, gridsize)
state_eigsh = eigenvectors_eigsh[:, 3].reshape(gridsize, gridsize)
#ax_states = fig_state4.add_subplot(1, 3, 1, projection='3d')
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_MG) * state_MG)
#ax_states.set_title("Multigrid")
#ax_states = fig_state4.add_subplot(1, 3, 2, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_eigsh) * state_eigsh)
#ax_states.set_title("Lanczos")
#ax_states = fig_state4.add_subplot(1, 3, 3, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_exact) * state_exact)
#ax_states.set_title("Exact")

ax_states = fig_state4.add_subplot(1, 3, 1)
ax_states.imshow(np.ma.conjugate(state_MG) * state_MG)
ax_states.set_title("Multigrid")
ax_states = fig_state4.add_subplot(1, 3, 2)
ax_states.imshow(np.ma.conjugate(state_eigsh) * state_eigsh)
ax_states.set_title("Lanczos")
ax_states = fig_state4.add_subplot(1, 3, 3)
ax_states.imshow(np.ma.conjugate(state_exact) * state_exact)
ax_states.set_title("Exact")


fig_state5 = plt.figure(5)
fig_state5.suptitle("State (1, 3)")
state_MG = eigenvectors_MG[:, 4].reshape(gridsize, gridsize)
state_exact = exact_eigenvectors[:, 4].reshape(gridsize, gridsize)
state_eigsh = eigenvectors_eigsh[:, 4].reshape(gridsize, gridsize)
#ax_states = fig_state5.add_subplot(1, 3, 1, projection='3d')
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_MG) * state_MG)
#ax_states.set_title("Multigrid")
#ax_states = fig_state5.add_subplot(1, 3, 2, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_eigsh) * state_eigsh)
#ax_states.set_title("Lanczos")
#ax_states = fig_state5.add_subplot(1, 3, 3, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_exact) * state_exact)
#ax_states.set_title("Exact")

ax_states = fig_state5.add_subplot(1, 3, 1)
ax_states.imshow(np.ma.conjugate(state_MG) * state_MG)
ax_states.set_title("Multigrid")
ax_states = fig_state5.add_subplot(1, 3, 2)
ax_states.imshow(np.ma.conjugate(state_eigsh) * state_eigsh)
ax_states.set_title("Lanczos")
ax_states = fig_state5.add_subplot(1, 3, 3)
ax_states.imshow(np.ma.conjugate(state_exact) * state_exact)
ax_states.set_title("Exact")

fig_state6 = plt.figure(6)
fig_state6.suptitle("State (3, 1)")
state_MG = eigenvectors_MG[:, 5].reshape(gridsize, gridsize)
state_exact = exact_eigenvectors[:, 5].reshape(gridsize, gridsize)
state_eigsh = eigenvectors_eigsh[:, 5].reshape(gridsize, gridsize)
#ax_states = fig_state6.add_subplot(1, 3, 1, projection='3d')
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_MG) * state_MG)
#ax_states.set_title("Multigrid")
#ax_states = fig_state6.add_subplot(1, 3, 2, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_eigsh) * state_eigsh)
#ax_states.set_title("Lanczos")
#ax_states = fig_state6.add_subplot(1, 3, 3, projection="3d")
#ax_states.plot_surface(X, Y, np.ma.conjugate(state_exact) * state_exact)
#ax_states.set_title("Exact")

ax_states = fig_state6.add_subplot(1, 3, 1)
ax_states.imshow(np.ma.conjugate(state_MG) * state_MG)
ax_states.set_title("Multigrid")
ax_states = fig_state6.add_subplot(1, 3, 2)
ax_states.imshow(np.ma.conjugate(state_eigsh) * state_eigsh)
ax_states.set_title("Lanczos")
ax_states = fig_state6.add_subplot(1, 3, 3)
ax_states.imshow(np.ma.conjugate(state_exact) * state_exact)
ax_states.set_title("Exact")

fig_error, axarray = plt.subplots(num_eigenvals / 2, 2, sharex='col', sharey='row')
row_index = 0
column_index = 0
for i in xrange(num_eigenvals):
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


fig_eigvals, axarray = plt.subplots(num_eigenvals / 2, 2, sharex='col', sharey='row')
row_index = 0
column_index = 0
for i in xrange(num_eigenvals):
    if i != 0:
        if i % 2 == 0:
            row_index += 1
            if column_index > 0:
                column_index -= 1
        else:
            column_index += 1
    axarray[row_index, column_index].plot(iter_x, eigenvalues[:, i])
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("Eigenvalue na V-cycles")
fig_eigvals.text(0.5, 0.04, 'x', ha='center', va='center')


plt.show()