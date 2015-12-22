import numpy as np
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTSolver import MGCMTSolver
import time
import matplotlib.pyplot as plt

# GIVES THE SAME RESULT AS vcycleTest.m

solver = MGCMTSolver()
stencil_maker = MGCMTStencilMaker()


gridsize = 2**4
A = stencil_maker.laplacian(gridsize)


f = np.zeros((gridsize, 1))

x0 = np.ones((gridsize, 1))

x = x0

x = solver.vcycle(x, f, A, stencil_maker, nu1=4, nu2=4)

print np.linalg.norm(x) # Expected value is 0.17756

plt.plot(x)
plt.show()