import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
import scipy.sparse.linalg as sparsela

stencil_maker = MGCMTStencilMaker()
solver = MGCMTSolver()
tolerance = 10**-4

############################# FINE GRID #####################################################
fine_gridsize = 2**6
fine_gridspacing = 1. / fine_gridsize
x = np.arange(0, 1, fine_gridspacing)
y = x
xx, yy = np.meshgrid(x, y)
fine_laplacian = stencil_maker.laplacian(fine_gridsize, dimension="2d")
fine_hamiltonian = (-1 / np.pi ** 2) * fine_laplacian
fine_eigvals, fine_eigvecs = sparsela.eigsh(fine_hamiltonian, which="SM")
print fine_eigvals
fine_state = fine_eigvecs[:, 0]
fine_state.shape = (len(fine_state), 1)
#############################################################################################

############################# COARSE GRID ###################################################
bad_gridsize = 2**4
bad_gridspacing = 1. / bad_gridsize
X = np.arange(0, 1, bad_gridspacing)
Y = X
XX, YY = np.meshgrid(X, Y)
bad_laplacian = stencil_maker.laplacian(bad_gridsize, dimension="2d")
bad_hamiltonian = (-1 / np.pi ** 2) * bad_laplacian
bad_eigvals, bad_eigvecs = sparsela.eigsh(bad_hamiltonian, which="SM")
print bad_eigvals
bad_state = bad_eigvecs[:, 0]
bad_state.shape = (len(bad_state), 1)
#############################################################################################

interpolation_matrix = stencil_maker.interpolation(bad_gridsize, fine_gridsize, dimension="2d")
restriction_matrix = stencil_maker.restriction(fine_gridsize, bad_gridsize, dimension="2d")


interpolated_state = interpolation_matrix * bad_state
restricted_state = restriction_matrix * fine_state


bad_state = bad_state.reshape(bad_gridsize, bad_gridsize)
fine_state = fine_state.reshape(fine_gridsize, fine_gridsize)
interpolated_state = interpolated_state.reshape(fine_gridsize, fine_gridsize)
restricted_state = restricted_state.reshape(bad_gridsize, bad_gridsize)


fig_bad_state = pl.figure(1)
axes_bad_state = fig_bad_state.gca(projection="3d")
axes_bad_state.plot_surface(XX, YY, np.ma.conjugate(bad_state.T) * bad_state)
pl.suptitle("Bad eigsh")

fig_restricted_state = pl.figure(2)
axes_restricted_state = fig_restricted_state.gca(projection="3d")
axes_restricted_state.plot_surface(XX, YY, np.ma.conjugate(restricted_state.T) * restricted_state)
pl.suptitle("Restricted")

fig_fine_state = pl.figure(3)
axes_fine_state = fig_fine_state.gca(projection="3d")
axes_fine_state.plot_surface(xx, yy, np.ma.conjugate(fine_state.T) * fine_state)
pl.suptitle("Fine eigsh")

fig_interpolated_state = pl.figure(4)
axes_interpolated_state = fig_interpolated_state.gca(projection="3d")
axes_interpolated_state.plot_surface(xx, yy, np.ma.conjugate(interpolated_state.T) * interpolated_state)
pl.suptitle("Interpolated")


pl.show()
