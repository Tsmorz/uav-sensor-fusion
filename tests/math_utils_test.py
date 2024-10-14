"""public doc string."""

import numpy as np
import pytest

from src.math_utils import matrix_exponential, skew


@pytest.mark.parametrize(
    "vector", [np.array([1, 2, 3]), np.array([[1], [2], [3]])]
)
def test_skew(vector):
    """Test the skew symmetric function."""
    # Arrange
    expected = np.array(
        [
            [0.0, -3.0, 2.0],
            [3.0, 0.0, -1.0],
            [-2.0, 1.0, 0.0],
        ]
    )

    # Act
    sk = skew(vector)

    # Assert
    np.testing.assert_equal(expected, sk)


@pytest.mark.parametrize("vector", [np.ones(4), np.ones(2)])
def test_skew_fails(vector):
    """Test the skew function fails with vectors not equal to three."""
    # Assert
    with pytest.raises(ValueError):
        skew(vector)


@pytest.mark.parametrize("t", [1.0, 0.01])
def test_matrix_exponential(t):
    """Test matrix exponential function with different time lengths."""
    # Arrange
    matrix = np.eye(3)
    expected = np.exp(t) * matrix

    # Act
    mat_exp = matrix_exponential(matrix, t=t)

    # Assert
    np.testing.assert_array_almost_equal(expected, mat_exp, decimal=3)


def test_matrix_exponential_fail():
    """Test that matrix exponential function fails with non-square matrices."""
    # Arrange
    matrix = np.ones((4, 3))

    # Assert
    with pytest.raises(ValueError):
        matrix_exponential(matrix)
