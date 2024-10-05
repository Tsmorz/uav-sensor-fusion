"""public doc string."""

import pytest


def sample_function(variable: int) -> int:
    """Sample function for testing.

    :param variable: variable passed into function.
    :return: variable passed in is passed out.
    """
    return variable


@pytest.mark.parametrize(
    ("variable", "expected_answer"), [(1, 1), (2, 2), (3, 3), (4, 4)]
)
def test_sample_function(variable: int, expected_answer: int):
    """Test sample function."""
    # Arrange

    # Act
    answer = sample_function(variable)

    # Assert
    assert answer == expected_answer
