import numpy as np
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
import time

## GIVES SAME RESULT AS MATLAB sorTest.m file


solver = MGCMTSolver()
stencil_maker = MGCMTStencilMaker()


gridsize = 2**4
A = stencil_maker.laplacian(gridsize)

f = np.zeros((gridsize, 1))

x0 = np.ones((gridsize, 1))

x = x0

for i in xrange(5):
    x = solver.sor(x, f, A, nu=4, omega=2. / 3.)

print np.linalg.norm(x)  # Expected value is 2.63327

