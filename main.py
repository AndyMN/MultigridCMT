import numpy as np
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
import scipy.sparse.linalg as sparsela
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time


solver = MGCMTSolver()
stencil_maker = MGCMTStencilMaker()
gridsize = 2**8
gridspacing = 1. / gridsize
x_axis = np.arange(0, 1, gridspacing)
machine_eps = np.finfo(float).eps
tolerance = 10 ** (-4)
laplacian = stencil_maker.laplacian(gridsize)
hamiltonian = (-1 / np.pi ** 2) * laplacian
num_eigenvalues = 6

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
print "Start eigs method"
time_start_eigs = time.clock()
eigenvalues_eigs, v = sparsela.eigsh(hamiltonian, which='SM', k=num_eigenvalues, tol=tolerance)
time_end_eigs = time.clock()
execution_time_eigs = time_end_eigs - time_start_eigs

eigenvectors_eigs = np.zeros((gridsize, num_eigenvalues))
for i, array in enumerate(v):
    for j in xrange(num_eigenvalues):
        eigenvectors_eigs[i, j] = array[j]


residu_eigs = np.zeros((1, num_eigenvalues))
for i in xrange(gridsize):
    for j in xrange(num_eigenvalues):
        residu_eigs[0, j] += (eigenvectors_eigs[i, j] - eigenvectors_exact[i, j]) ** 2


eigenvalue_diff_eigs = [abs(eigenvalues_eigs[i] - eigenvalues_exact[i]) for i in xrange(num_eigenvalues)]
print "End eigs method"
###########################################


#Shifted power method using eigsh bad grid values as guess
###########################################################
print "Start shift method"
bad_gridsize = 2**5
bad_laplacian = stencil_maker.laplacian(bad_gridsize)
bad_gridspacing = 1. / bad_gridsize
bad_hamiltonian = (-1 / np.pi ** 2) * bad_laplacian

iterations_shift = np.zeros((1, num_eigenvalues))
eigenvalues_shift = np.zeros((1, num_eigenvalues))
eigenvectors_shift = np.zeros((gridsize, num_eigenvalues))

time_start_shift = time.clock()
bad_w, bad_v = sparsela.eigsh(bad_hamiltonian, which='SM', k=num_eigenvalues, tol=tolerance)
print bad_w
bad_eigenvec_shift = np.zeros((bad_gridsize, num_eigenvalues))
for i, array in enumerate(bad_v):
    for j in xrange(num_eigenvalues):
        bad_eigenvec_shift[i, j] = array[j]

for i in xrange(num_eigenvalues):
    print i
    v_guess_shift = solver.interpolate(bad_eigenvec_shift[:, i], stencil_maker, gridsize)
    v_start_shift = v_guess_shift / np.linalg.norm(v_guess_shift)
    v_new_shift = v_start_shift

    mu_matrix = sparse.eye(gridsize, gridsize) * bad_w[i]

    shifted_matrix = hamiltonian - mu_matrix

    #base_norm_shift = np.linalg.norm(v_new_shift - hamiltonian.dot(np.zeros((gridsize, 1))))

    iteration_number_shift = 0
    eigenvalue_shift = 0
    eigenvalue_calc_shift_1 = 0
    eigenvalue_calc_shift_2 = 0
    v_new_shift_2 = np.zeros((gridsize, 1))
    eigenvalue_relativediff_shift = 10
    while eigenvalue_relativediff_shift > tolerance:
        temp_vec = solver.vcycle(np.zeros((gridsize, 1)), v_new_shift, shifted_matrix, stencil_maker, nu1=20, nu2=20)
        res = np.linalg.norm(v_new_shift - shifted_matrix * temp_vec)
        print res
        #temp_vec = solver.wjacobi(np.zeros((gridsize, 1)), v_new_shift, shifted_matrix)
        #temp_vec = sparsela.spsolve(shifted_matrix, v_new_shift)
        v_new_shift_2 = v_new_shift
        v_new_shift = temp_vec / np.linalg.norm(temp_vec)
        # v^T * A * v
        eigenvalue_shift = np.dot(np.dot(np.transpose(v_new_shift), hamiltonian.toarray()), v_new_shift)
        eigenvalue_calc_shift_2 = eigenvalue_calc_shift_1
        eigenvalue_calc_shift_1 = eigenvalue_shift
        eigenvalue_relativediff_shift = abs(eigenvalue_calc_shift_1 - eigenvalue_calc_shift_2)
     #   print eigenvalue_relativediff_shift
        #relative_error = np.linalg.norm(v_new_shift - hamiltonian.dot(temp_vec) / base_norm_shift)
        #print relative_error

        delta_v_new_shift = np.ma.conjugate(v_new_shift) * v_new_shift - np.ma.conjugate(v_new_shift_2) * v_new_shift_2
        error = np.linalg.norm(delta_v_new_shift)
        #print np.linalg.norm(delta_v_new_shift)

        iteration_number_shift += 1

    iterations_shift[0, i] = iteration_number_shift
    eigenvalues_shift[0, i] = eigenvalue_shift
    print eigenvalue_shift
    eigenvectors_shift[:, i] = v_new_shift[:, 0]  # For iterative
    #eigenvectors_shift[:, i] = v_new_shift  # For spsolve

time_end_shift = time.clock()

execution_time_shift = time_end_shift - time_start_shift

residu_shift = np.zeros((1, num_eigenvalues))
for i in xrange(gridsize):
    for j in xrange(num_eigenvalues):
        residu_shift[0, j] += (eigenvectors_shift[i, j] - eigenvectors_exact[i, j]) ** 2


eigenvalue_diff_shift = [abs(eigenvalues_shift[0, i] - eigenvalues_exact[i]) for i in xrange(num_eigenvalues)]
print "End shift method"
###########################################################


# Rayleigh-quotient-iteratie
###########################################################
print "Start RQI method"
bad_gridsize = 2**5
bad_laplacian = stencil_maker.laplacian(bad_gridsize)
bad_gridspacing = 1. / bad_gridsize
bad_hamiltonian = (-1 / np.pi ** 2) * bad_laplacian

iterations_RQI = np.zeros((1, num_eigenvalues))
eigenvalues_RQI = np.zeros((1, num_eigenvalues))
eigenvectors_RQI = np.zeros((gridsize, num_eigenvalues))

time_start_RQI = time.clock()
bad_w_RQI, bad_v_RQI = sparsela.eigsh(bad_hamiltonian, which='SM', k=num_eigenvalues, tol=tolerance)
print bad_w_RQI
bad_eigenvec_RQI = np.zeros((bad_gridsize, num_eigenvalues))
for i, array in enumerate(bad_v_RQI):
    for j in xrange(num_eigenvalues):
        bad_eigenvec_RQI[i, j] = array[j]

for i in xrange(num_eigenvalues):
    print i
    v_guess_RQI = solver.interpolate(bad_eigenvec_RQI[:, i], stencil_maker, gridsize)
    v_start_RQI = v_guess_RQI / np.linalg.norm(v_guess_RQI)
    eigenvalue_start_RQI = np.dot(np.dot(np.transpose(v_start_RQI), hamiltonian.toarray()), v_start_RQI)

    v_new_RQI = v_start_RQI
    eigenvalue_new_RQI = eigenvalue_start_RQI
    eigenvalue_calc_RQI_1 = 0
    eigenvalue_calc_RQI_2 = 0
    iteration_number_RQI = 0
    is_converged_RQI = False
    while not is_converged_RQI:
        shifted_matrix_RQI = hamiltonian - sparse.eye(gridsize, gridsize) * eigenvalue_new_RQI
        temp_vec = sparsela.spsolve(shifted_matrix_RQI, v_new_RQI)
        v_new_RQI = temp_vec / np.linalg.norm(temp_vec)
        eigenvalue_new_RQI = np.dot(np.dot(np.transpose(v_new_RQI), hamiltonian.toarray()), v_new_RQI)
        # Check if eigenvalues are converged (same precision as eigsh)
        eigenvalue_calc_RQI_2 = eigenvalue_calc_RQI_1
        eigenvalue_calc_RQI_1 = eigenvalue_new_RQI
        eigenvalue_relativediff_RQI = eigenvalue_calc_RQI_1 - eigenvalue_calc_RQI_2
        if abs(eigenvalue_relativediff_RQI) <= tolerance:
            is_converged_RQI = True
        else:
            is_converged_RQI = False
        iteration_number_RQI += 1

    iterations_RQI[0, i] = iteration_number_RQI
    eigenvalues_RQI[0, i] = eigenvalue_new_RQI
    eigenvectors_RQI[:, i] = v_new_RQI

time_end_RQI = time.clock()
execution_time_RQI = time_end_RQI - time_start_RQI

residu_RQI = np.zeros((1, num_eigenvalues))
for i in xrange(gridsize):
    for j in xrange(num_eigenvalues):
        residu_RQI[0, j] += (eigenvectors_RQI[i, j] - eigenvectors_exact[i, j]) ** 2

eigenvalue_diff_RQI = [abs(eigenvalues_RQI[0, i] - eigenvalues_exact[i]) for i in xrange(num_eigenvalues)]
print "End RQI method"
###########################################################
print "Iterations Shift Method: %s" % iterations_shift
print "Iterations RQI Method: %s" % iterations_RQI
print "\n"
print "Eigenvalue Exact: %s" % eigenvalues_exact
print "Eigenvalue Eigs: %s" % eigenvalues_eigs
print "Eigenvalue Shift: %s" % eigenvalues_shift
print "Eigenvalue RQI: %s" % eigenvalues_RQI
print "\n"
print "Residuals Eigs Method: %s" % residu_eigs
print "Residuals Shift Method: %s" % residu_shift
print "Residuals RQI Method: %s" % residu_RQI
print "\n"
print "Absolute Diff Eigs Method: %s" % eigenvalue_diff_eigs
print "Absolute Diff Shift Method: %s" % eigenvalue_diff_shift
print "Absolute Diff RQI Method: %s" % eigenvalue_diff_RQI
print "\n"
print "Execution Time Eigs Method: %f" % execution_time_eigs
print "Execution Time Shift Method: %f" % execution_time_shift
print "Execution Time RQI Method: %f" % execution_time_RQI

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
plt.suptitle("Exact Method")
fig_exact.text(0.5, 0.04, 'x', ha='center', va='center')
fig_exact.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


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


fig_RQI, axarray = plt.subplots(num_eigenvalues / 2, 2, sharex='col', sharey='row')
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
    psi_sq = np.ma.conjugate(eigenvectors_RQI[:, i]) * eigenvectors_RQI[:, i]
    axarray[row_index, column_index].plot(x_axis, psi_sq, c=np.random.rand(3,))
    axarray[row_index, column_index].set_title("n = " + str(i+1))
plt.suptitle("RQI Method")
fig_RQI.text(0.5, 0.04, 'x', ha='center', va='center')
fig_RQI.text(0.06, 0.5, '|Psi(x)|^2', ha='center', va='center', rotation='vertical')


plt.show()
