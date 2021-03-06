import numpy as np
import scipy.sparse as spsparse
import math

class MGCMTStencilMaker:
    """
    Klasse die al de gebruikte operators produceerd.
    """

    def __init__(self):
        pass



    def laplacian(self, n, dimension="1d"):
        laplacian = None
        h = 1. / n

        if dimension == "1d":
            laplacian = spsparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format="csc")
            laplacian *= (1 / h**2)
        elif dimension == "2d":
            laplacian_1D = self.laplacian(n, dimension="1d")
            laplacian = spsparse.kronsum(laplacian_1D, laplacian_1D)
        return laplacian

    def interpolation(self, old_gridsize, new_gridsize, dimension="1d"):
        new_gridsize = int(new_gridsize) # Need to make sure it's an int for the slicing indices
        if dimension == "1d":
            # General form of the interpolation matrix that goes from a grid 2^p to a grid 2^n with n > p and n and p integer
            old_gridsize_poweroftwo = math.log(old_gridsize) / math.log(2)
            new_gridsize_poweroftwo = math.log(new_gridsize) / math.log(2)
            if new_gridsize_poweroftwo > old_gridsize_poweroftwo:
                if old_gridsize_poweroftwo.is_integer():
                    if new_gridsize_poweroftwo.is_integer():
                        maindiag_element = int(new_gridsize / old_gridsize)
                        power_diff = new_gridsize_poweroftwo - old_gridsize_poweroftwo
                        prefactor = (1. / 2) ** power_diff
                        diag_elements = [i + 1 if i + 1 <= maindiag_element else maindiag_element - (i + 1 - maindiag_element) for i in xrange(2*maindiag_element - 1)]
                        diagonals = [(i + 1) - maindiag_element if i + 1 < maindiag_element else 0 if i + 1 == maindiag_element else (i + 1) - maindiag_element for i in xrange(2*maindiag_element - 1)]
                        interpolation_matrix = spsparse.diags(diag_elements, diagonals, shape=(new_gridsize, new_gridsize), format="csc")
                        interpolation_matrix = prefactor * interpolation_matrix[:, maindiag_element-1:new_gridsize:maindiag_element]
                        return interpolation_matrix
                    else:
                        print "New gridsize isn't a power of 2 !"
                else:
                    print "Old gridsize isn't a power of 2 !"
            else:
                print "New gridsize isn't bigger than old gridsize !"
        elif dimension == "2d":
            S = self.interpolation(old_gridsize, new_gridsize, dimension="1d")

            interpolation_matrix = spsparse.kron(S, S, format="csc")
            return interpolation_matrix


    def restriction(self, old_gridsize, new_gridsize, dimension="1d"):
        if dimension == "1d":
            old_gridsize_poweroftwo = math.log(old_gridsize) / math.log(2)
            new_gridsize_poweroftwo = math.log(new_gridsize) / math.log(2)
            if new_gridsize_poweroftwo < old_gridsize_poweroftwo:
                if old_gridsize_poweroftwo.is_integer():
                    if new_gridsize_poweroftwo.is_integer():
                        power_diff = old_gridsize_poweroftwo - new_gridsize_poweroftwo
                        prefactor = (1. / 2) ** power_diff
                        interpolation_matrix = self.interpolation(new_gridsize, old_gridsize)
                        restriction_matrix = prefactor * np.transpose(interpolation_matrix)
                        return spsparse.csc_matrix(restriction_matrix)
                    else:
                        print "New gridsize isn't a power of 2 !"
                else:
                    print "Old gridsize isn't a power of 2 !"
            else:
                print "New gridsize is bigger (more elements) than old gridsize !"
        elif dimension == "2d":
            interpolation_matrix = self.interpolation(new_gridsize, old_gridsize, dimension="2d")
            restriction_matrix = 1. / 4. * np.transpose(interpolation_matrix)
            return restriction_matrix