import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import eig
from MGCMTStencilMaker import MGCMTStencilMaker
from MGCMTProcessor import MGCMTProcessor

class MGCMTSolver:
    """
    Klasse die alle opossingsmethoden bevat die getest zijn.
    """

    def __init__(self):
        self.stencil_maker = MGCMTStencilMaker()
        self.processor = MGCMTProcessor()

    def rqmin(self, A, v0, M = None, nu=4):
        x = np.array(v0)
        rho = np.dot(x.conj().T, A.dot(x)) / np.dot(x.conj().T, M.dot(x))

        gold = np.array(x)
        #print np.shape(M.dot(x))
        g = 2 * (A.dot(x) - rho * M.dot(x))
        #print np.shape(g)
        #g.shape = (len(x), 1)
        p = np.array(x)

        R = np.zeros((2, 2))
        RM = np.zeros((2, 2))

        k = 1
        for i in xrange(nu):
            if k == 1:
                p = np.array(-g)
            else:
                p = np.array(-g) + (np.dot(g.conj().T, M.dot(g)) / np.dot(gold.conj().T, M.dot(gold))) * p

            R[0, 0] = np.dot(x.conj().T, A.dot(x))
            R[0, 1] = np.dot(x.conj().T, A.dot(p))
            R[1, 0] = np.dot(p.conj().T, A.dot(x))
            R[1, 1] = np.dot(p.conj().T, A.dot(p))

            RM[0, 0] = np.dot(x.conj().T, M.dot(x))
            RM[0, 1] = np.dot(x.conj().T, M.dot(p))
            RM[1, 0] = np.dot(p.conj().T, M.dot(x))
            RM[1, 1] = np.dot(p.conj().T, M.dot(p))

            Reigvals, Reigvecs = eig(R, b=RM)
            Rx = np.array(Reigvecs[:, np.argmin(Reigvals)])
            delta = Rx[1] / Rx[0]
            x = np.array(x + delta * p)
            rho = np.dot(x.conj().T, A.dot(x)) / np.dot(x.conj().T, M.dot(x))
            gold = np.array(g)
            g = np.array(2 * (A.dot(x) - rho * M.dot(x)))
            k += 1

        return x, rho

    def vcycle_rqmg2(self, x_matrix, A, M, nu1=4, nu2=4, nmin=2, level=0):

        k = np.array(x_matrix)

        n = np.shape(k)[0]
        num_vecs = np.shape(k)[1]

        for i in xrange(num_vecs):
            k[:, i], rho = self.rqmin(A, k[:, i], M, nu=nu1)

        if level == 0:
            for nu in xrange(4):
                k = self.processor.gramschmidt(k)

        if n > nmin:

            interpolation_matrix = self.stencil_maker.interpolation(n / 2, n)
            restriction_matrix = self.stencil_maker.restriction(n, n / 2)

            A_coarse = restriction_matrix * A * interpolation_matrix
            M_coarse = restriction_matrix * M * interpolation_matrix

            k_coarse = np.zeros((n / 2, num_vecs))

            for i in xrange(num_vecs):
                k_coarse[:, i] = restriction_matrix * k[:, i]

            c = self.vcycle_rqmg2(k_coarse, A_coarse, M_coarse, nu1=nu1, nu2=nu2, nmin=nmin, level=level + 1)

            for i in xrange(num_vecs):
                c_fine = interpolation_matrix * c[:, i]
                k[:, i] = k[:, i] + c_fine
                k[:, i], rho = self.rqmin(A, k[:, i], M, nu=nu2)


        return k




    def vcycle_rqmg(self, x, A, M, nu1=4, nu2=4, nmin=2):
        k = np.array(x)
        n = len(k)

        k, rho = self.rqmin(A, k, M, nu=nu1)

        if n > nmin:

            interpolation_matrix = self.stencil_maker.interpolation(n / 2, n)
            restriction_matrix = self.stencil_maker.restriction(n, n / 2)

            A_coarse = restriction_matrix * A * interpolation_matrix
            M_coarse = restriction_matrix * M * interpolation_matrix

            k_coarse = restriction_matrix * k

            c, rho = self.vcycle_rqmg(k_coarse, A_coarse, M_coarse, nu1=nu1, nu2=nu2, nmin=nmin)
            c_fine = interpolation_matrix * c

            k = k + c_fine

            k, rho = self.rqmin(A, k, M, nu=nu2)

        return k, rho




    def twogridrqmin(self, A, v0, M, nu1=4, nu2=4):
        x = np.array(v0)

        n = len(x)

        x, l = self.rqmin(A, x, M, nu=nu1)

        restriction_matrix = self.stencil_maker.restriction(n, n / 2)
        interpolation_matrix = self.stencil_maker.interpolation(n / 2, n)

        A_coarse = restriction_matrix * A * interpolation_matrix
        M_coarse = restriction_matrix * M * interpolation_matrix

        x_coarse = restriction_matrix * x
        rho_coarse = np.dot(x_coarse.conj().T, A_coarse.dot(x_coarse)) / np.dot(x_coarse.conj().T, M_coarse.dot(x_coarse))

        gold = np.array(x_coarse)
        g = 2 * (A_coarse.dot(x_coarse) - rho_coarse * M_coarse.dot(x_coarse))
        p = np.array(x_coarse)

        R = np.zeros((2, 2))
        RM = np.zeros((2, 2))

        k = 1
        for i in xrange(nu1):
            if k == 1:
                p = np.array(-g)
            else:
                p = np.array(-g) + (np.dot(g.conj().T, M_coarse.dot(g)) / np.dot(gold.conj().T, M_coarse.dot(gold))) * p

            R[0, 0] = np.dot(x_coarse.conj().T, A_coarse.dot(x_coarse))
            R[0, 1] = np.dot(x_coarse.conj().T, A_coarse.dot(p))
            R[1, 0] = np.dot(p.conj().T, A_coarse.dot(x_coarse))
            R[1, 1] = np.dot(p.conj().T, A_coarse.dot(p))

            RM[0, 0] = np.dot(x_coarse.conj().T, M_coarse.dot(x_coarse))
            RM[0, 1] = np.dot(x_coarse.conj().T, M_coarse.dot(p))
            RM[1, 0] = np.dot(p.conj().T, M_coarse.dot(x_coarse))
            RM[1, 1] = np.dot(p.conj().T, M_coarse.dot(p))

            Reigvals, Reigvecs = eigh(R, b=RM)
            Rx = np.array(Reigvecs[:, np.argmin(Reigvals)])
            delta = Rx[1] / Rx[0]
            x = np.array(x + delta * interpolation_matrix * p)
            rho_coarse = np.dot(x_coarse.conj().T, A_coarse.dot(x_coarse)) / np.dot(x_coarse.conj().T, M_coarse.dot(x_coarse))
            gold = np.array(g)
            g = np.array(2 * (A_coarse.dot(x_coarse) - rho_coarse * M_coarse.dot(x_coarse)))
            k += 1

        x, l = self.rqmin(A, x, M, nu=nu2)

        return x, l



    def wjacobi(self, v0, f, A, nu=4, omega=2./3.):
        #  VEEL SHAPE SHENANIGANS OMDAT SPSOLVE NDARRAYS CAST NAAR DEFMATRIX EN DAN IS ALLES KAPOET

        n = len(v0)

        if f.shape is not (n, 1):
            f.shape = (n, 1)

        if v0.shape is not (n, 1):
            v0.shape = (n, 1)

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
    """
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

    """

    def vcycle(self, v0, f, A, stencil_maker, nu1=4, nu2=4, smoother=None, shift=0, lowest_level=2, dimension="1d"):
        if smoother is None:
            smoother = self.wjacobi

        n = len(v0)

        mu_matrix = sparse.eye(n) * shift
        shifted_matrix = A - mu_matrix  # Het is nodig om de shift en de operator A gescheiden te houden omdat we anders een foute coarse operator bekomen

        grid_dimension = 0
        if dimension == "1d":
            grid_dimension = n
        elif dimension == "2d":
            grid_dimension = np.sqrt(n)

        # reshape shenanigans because 1D numpy arrays have shape (n, 1) while 1D python arrays have shape (n)
        if f.shape is not (n, 1):
            f.shape = (n, 1)
        if v0.shape is not (n, 1):
            v0.shape = (n, 1)


        if grid_dimension < 2:
            print "Length of start vector is not a power of 2"
        elif grid_dimension == lowest_level:
            v = spsolve(shifted_matrix, f)
            v.shape = (len(v), 1)
            return v
        else:
            restriction_matrix = stencil_maker.restriction(grid_dimension, grid_dimension / 2, dimension=dimension)
            interpolation_matrix = stencil_maker.interpolation(grid_dimension / 2, grid_dimension, dimension=dimension)

            v = smoother(v0, f, shifted_matrix, nu=nu1)

            r = restriction_matrix * (f - shifted_matrix * v)
            e2h = np.zeros((np.shape(r)))

            coarse_operator = restriction_matrix * A * interpolation_matrix

            e2h = self.vcycle(e2h, r, coarse_operator, stencil_maker, shift=shift, smoother=smoother, lowest_level=lowest_level, dimension=dimension)
            e2h.shape = (len(e2h), 1)  # reshaping this because we are returning the first column in vcycle to make the life of the user easier

            e = interpolation_matrix * e2h
            v = v + e

            v = smoother(v, f, shifted_matrix, nu=nu2)

            # returning the first column so that the user doesn't need to extract it
            return v[:, 0]

    def twogrid(self, v0, f, A, stencil_maker, nu1=4, nu2=4, smoother=None, shift=0, dimension="1d"):
        if smoother is None:
            smoother = self.wjacobi

        n = len(v0)

        grid_dimension = 0
        if dimension == "1d":
            grid_dimension = n
        elif dimension == "2d":
            grid_dimension = np.sqrt(n)

        # reshape shenanigans because 1D numpy arrays have shape (n, 1) while 1D python arrays have shape (n)
        if f.shape is not (n, 1):
            f.shape = (n, 1)
        if v0.shape is not (n, 1):
            v0.shape = (n, 1)

        mu_matrix = sparse.eye(n) * shift
        coarse_mu_matrix = sparse.eye((n/2)) * shift
        shifted_matrix = A - mu_matrix

        restriction_matrix = stencil_maker.restriction(grid_dimension, grid_dimension / 2, dimension=dimension)
        interpolation_matrix = stencil_maker.interpolation(grid_dimension / 2, grid_dimension, dimension=dimension)

        coarse_operator = restriction_matrix * A * interpolation_matrix - coarse_mu_matrix

        v = smoother(v0, f, shifted_matrix, nu1)  # Pre smoothing

        r = restriction_matrix * (f - shifted_matrix * v)  # Restriction of fine grid residual

        e = spsolve(coarse_operator, r)  # Solve error equation
        e.shape = (len(e), 1)

        e = interpolation_matrix * e  # Interpolate error to fine grid

        v = v + e  # Correct fine grid approximation

        v = smoother(v, f, shifted_matrix, nu2)

        return v[:, 0]



    def vcycle_matrix(self, v0_matrix, f_matrix, A, stencil_maker, nu1=4, nu2=4, smoother=None, shifts=None, lowest_level=2, dimension="1d"):
        n = len(v0_matrix[:, 0])
        num_vectors = f_matrix.shape[1]

        if smoother is None:
            smoother = self.wjacobi

        if shifts is None:
            shifts = np.zeros((1, num_vectors))

        shifted_matrices = []
        for shift in shifts.T:
            shifted_matrices.append(A - (sparse.eye(n) * shift))
        shifted_matrices = np.array(shifted_matrices)

        grid_dimension = 0
        r_dimension = 0
        if dimension == "1d":
            grid_dimension = n
            r_dimension = n / 2
        elif dimension == "2d":
            grid_dimension = np.sqrt(n)
            r_dimension = n / 4


        v = np.zeros((n, num_vectors))
        r = np.zeros((r_dimension, num_vectors))
        e = np.zeros((n, num_vectors))

        if grid_dimension < 2:
            print "Length of start vector is not a power of 2"
        elif grid_dimension == lowest_level:
            for i in xrange(num_vectors):
                v[:, i] = spsolve(shifted_matrices[i], f_matrix[:, i])
            return v
        else:
            restriction_matrix = stencil_maker.restriction(grid_dimension, grid_dimension / 2, dimension=dimension)

            interpolation_matrix = stencil_maker.interpolation(grid_dimension / 2, grid_dimension, dimension=dimension)

            for i in xrange(num_vectors):
                v[:, i] = smoother(v0_matrix[:, i], f_matrix[:, i], shifted_matrices[i], nu=nu1)[:, 0]


            for i in xrange(num_vectors):
                r[:, i] = restriction_matrix * (f_matrix[:, i] - shifted_matrices[i] * v[:, i])

            e2h = np.zeros((np.shape(r)))

            coarse_operator = restriction_matrix * A * interpolation_matrix

            e2h = self.vcycle_matrix(e2h, r, coarse_operator, stencil_maker, shifts=shifts, smoother=smoother, lowest_level=lowest_level, dimension=dimension)
            #e2h.shape = (len(e2h), 1)  # reshaping this because we are returning the first column in vcycle to make the life of the user easier

            for i in xrange(num_vectors):
                e[:, i] = interpolation_matrix * e2h[:, i]
                v[:, i] = v[:, i] + e[:, i]
                v[:, i] = smoother(v[:, i], f_matrix[:, i], shifted_matrices[i], nu=nu2)[:, 0]

            v = self.processor.gramschmidt(v)

            return v