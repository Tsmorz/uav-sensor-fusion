"""public doc string."""

import numpy as np
import numpy.linalg as LA


def skew(vector: np.ndarray) -> np.ndarray:
    """
    Calculate the skew symmetric matrix from a given vector.

    :param vector: A 3D vector represented as a numpy array.
    :return: The skew symmetric matrix of the given vector.
    """
    dim = len(np.shape(vector))
    if dim == 2:
        vector = np.reshape(vector, (3,))
    if len(vector) != 3:
        raise ValueError("Input vector must have a dimension of 3 or less.")

    sk = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    return sk


def matrix_exponential(matrix: np.ndarray, t: float = 1.0) -> np.ndarray:
    """
    Calculate the matrix exponential of a given matrix.

    :param matrix: A square matrix represented as a numpy array.
    :param t: The time parameter.
    :return: The matrix exponential of the given matrix.
    """
    if np.shape(matrix)[0] != np.shape(matrix)[1]:
        raise ValueError("Input matrix must be square.")

    num = np.shape(matrix)[0]
    val, vec = LA.eig(matrix)

    s = np.eye(num)
    matrax_exp = vec @ (np.exp(val * t) * s) @ LA.inv(vec)
    return matrax_exp.real


def main():
    """Test the dynamics functions within this module."""
    return


if __name__ == "__main__":
    main()
