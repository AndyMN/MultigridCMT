import numpy as np


class MGCMTProcessor:


    def __init__(self):
        pass

    def projection(self, v, u):
        """
        Projection operator. Projects v onto u
        :param v: vector to project
        :param u: vector on which to project
        :return: projected vector
        """
        inner1 = float(np.inner(v, u))
        inner2 = float(np.inner(u, u))

        return (inner1 / inner2) * u

    def gramschmidt(self, vectors, modified=1):
        """
        Gram-Schmidt OrthoNormalization
        :param vectors: matrix with the columns representing the vectors to orthonormalize
        :param modified: 0 is Classical Gram-Schmidt (numerically unstable),  1 is Modified Gram-Schmidt (numerically stable)
        :return: matrix with columns representing orthonormal vectors
        """
        columns = vectors.shape[1]
        rows = vectors.shape[0]
        orthogonal_vectors = np.zeros((rows, columns))
        orthonormal_vectors = np.zeros((rows, columns))

        if not modified:
            for j, column in enumerate(vectors.T):
                if j == 0:
                    orthogonal_vectors[:, j] = column
                else:
                    orthogonal_vectors[:, j] = column
                    for i in xrange(j):
                        orthogonal_vectors[:, j] = orthogonal_vectors[:, j] - self.projection(column, orthogonal_vectors[:, i])
            return self.normalize(orthogonal_vectors)
        else:
            for i, column in enumerate(vectors.T):
                orthogonal_vectors[:, i] = column
            for i, vector in enumerate(orthogonal_vectors.T):
                orthonormal_vectors[:, i] = orthogonal_vectors[:, i] / np.linalg.norm(orthogonal_vectors[:, i])
                for j in xrange(i + 1, columns):
                    orthogonal_vectors[:, j] = orthogonal_vectors[:, j] - self.projection(orthogonal_vectors[:, j], orthonormal_vectors[:, i])
            return orthonormal_vectors

    def normalize(self, vectors):
        """
        Normalizes vectors
        :param vectors: matrix with columns representing the vectors to normalize
        :return: matrix with columns representing the normalized vectors
        """
        normalized_vectors = np.zeros((vectors.shape[0], vectors.shape[1]))

        for j, vector in enumerate(vectors.T):
            normalized_vectors[:, j] = vector / np.linalg.norm(vector)

        return normalized_vectors

    def orthogonality_check(self, vectors):
        columns = vectors.shape[1]
        orthogonality_matrix = np.zeros((columns, columns))

        for i in xrange(columns):
            for j in xrange(columns):
                orthogonality_matrix[i, j] = np.inner(vectors[:, i], vectors[:, j])

        return orthogonality_matrix