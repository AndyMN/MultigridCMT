import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import math

class MGCMTSolver:

    def __init__(self):
        pass

    def wjacobi(self, v0, f, A, nu=4, omega=2./3.):

        #  VEEL SHAPE SHENANIGANS OMDAT SPSOLVE NDARRAYS CAST NAAR DEFMATRIX EN DAN IS ALLES KAPOET

        n = len(v0)
        D = sparse.diags(A.diagonal(), 0, shape=(n, n), format="csc")
        I = sparse.eye(n, format="csc")
        Rwj = (I - omega * spsolve(D, A))
        Rwj.shape = (Rwj.shape[0], Rwj.shape[1])

        v = v0
        for i in xrange(nu):
            v = Rwj * v
            result = spsolve(D, f)
            result.shape = (len(result), 1)

            v = v + omega * result

        return v

    def gseidel(self, v0, f, A, nu=4):
        n = len(v0)
        D = sparse.diags(A.diagonal(), 0, shape=(n, n), format="csc")
        L = -sparse.tril(A, -1)
        U = -sparse.triu(A, 1)
        Rg = spsolve(D-L, U)
        Rg.shape = (Rg.shape[0], Rg.shape[1])

        v = v0
        for i in xrange(nu):
            v = Rg * v

            result = spsolve(D-L, f)
            result.shape = (len(result), 1)

            v = v + result

        return v

    def sor(self, v0, f, A, nu=4, omega=1):
        n = len(v0)
        D = sparse.diags(A.diagonal(), 0, shape=(n, n), format="csc")
        L = -sparse.tril(A, -1)
        U = -sparse.triu(A, 1)
        Rs = spsolve(D - omega * L, (1 - omega) * D + omega * U)
        Rs.shape = (Rs.shape[0], Rs.shape[1])

        v = v0
        for i in xrange(nu):
            v = Rs * v

            result = spsolve(D-L, f)
            result.shape = (len(result), 1)

            v = v + omega * result

        return v

# TODO: FIX GSEIDELRB
    def gseidelrb(self, v0, f, A, nu=4):
        n = len(v0)
        I = sparse.eye(n)
        h = 1. / n

        Dinv = spsolve(sparse.diags(A.diagonal(), 0, shape=(n, n), format="csc"), I)
        Dinv = Dinv / h ** 2
        Dinvr = I.toarray()
        Dinvb = I.toarray()
        for i in xrange(n):
            if i % 2 != 0:
                Dinvr[i, i] = Dinv[i, i]
            else:
                Dinvb[i, i] = Dinv[i, i]
        Dinvr = sparse.csc_matrix(Dinvr)
        Dinvb = sparse.csc_matrix(Dinvb)

        ee = np.transpose(np.mod(np.array([i+1 for i in xrange(n)]), 2))
        not_ee = np.array([int(elem) for elem in np.logical_not(ee)])
        Rr = sparse.diags([ee, ee, ee], [-1, 0, 1], shape=(n, n), format="csc")
        Rb = sparse.diags([not_ee, not_ee, not_ee], [-1, 0, 1], shape=(n,n), format="csc")

        v = v0
        for i in xrange(nu):
            v = Dinvr * (Rr * v + not_ee * (h ** 2) * f)
            v = Dinvb * (Rb * v + ee * (h ** 2) * f)

        return v



    def interpolate(self, v, stencil_maker, new_gridsize):
        old_gridsize = len(v)
        interpolation_matrix = stencil_maker.interpolation(old_gridsize, new_gridsize)
        interpolated_v = interpolation_matrix * v
        interpolated_v.shape = (len(interpolated_v), 1)
        return interpolated_v

    def restrict(self, v, stencil_maker, new_gridsize):
        old_gridsize = len(v)
        restriction_matrix = stencil_maker.restriction(old_gridsize, new_gridsize)
        restricted_v = restriction_matrix * v
        restricted_v.shape = (len(restricted_v), 1)
        return restricted_v

    def vcycle(self, v0, f, A, stencil_maker, nu1=4, nu2=4, smoother=None, shift=0):
        if smoother == None:
            smoother = self.wjacobi

        n = len(v0)

        mu_matrix = sparse.eye(n) * shift
        shifted_matrix = A - mu_matrix

        if n < 2:
            print "Length of start vector is not a power of 2"
        elif n == 16:
            v = spsolve(shifted_matrix, f)
            v.shape = (len(v), 1)
            return v
        else:
            restriction_matrix = stencil_maker.restriction(n, n / 2)
            interpolation_matrix = stencil_maker.interpolation(n / 2, n)

            v = smoother(v0, f, shifted_matrix, nu=nu1)

            r = restriction_matrix * (f - shifted_matrix * v)
            e2h = np.zeros((np.shape(r)))

            coarse_operator = restriction_matrix * A * interpolation_matrix

            e2h = self.vcycle(e2h, r, coarse_operator, stencil_maker, shift=shift, smoother=smoother)
            e = interpolation_matrix * e2h
            v = v + e

            v = smoother(v, f, shifted_matrix, nu=nu2)

            return v

    def twogrid(self, v0, f, A, stencil_maker, nu1=4, nu2=4, smoother=None, shift=0):
        if smoother == None:
            smoother = self.wjacobi

        n = len(v0)
        mu_matrix = sparse.eye(n) * shift
        coarse_mu_matrix = sparse.eye((n/2)) * shift
        shifted_matrix = A - mu_matrix

        restriction_matrix = stencil_maker.restriction(n, n / 2)
        interpolation_matrix = stencil_maker.interpolation(n / 2, n)

        coarse_operator = restriction_matrix * A * interpolation_matrix - coarse_mu_matrix

        v = smoother(v0, f, shifted_matrix, nu1)  # Pre smoothing

        r = restriction_matrix * (f - shifted_matrix * v)  # Restriction of fine grid residual

        e = spsolve(coarse_operator, r)  # Solve error equation
        e.shape = (len(e), 1)

        e = interpolation_matrix * e  # Interpolate error to fine grid

        v = v + e  # Correct fine grid approximation

        v = smoother(v, f, shifted_matrix, nu2)

        return v

