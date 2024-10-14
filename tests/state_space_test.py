import numpy as np
import pytest

from src.state_space import StateSpace


def test_state_space():
    """Test the state space class."""
    # Arrange
    expected_state_space_a = np.eye(2)
    expected_state_space_b = np.eye(2)

    A = np.eye(2)
    B = np.eye(2)

    # Act
    state_space = StateSpace(A, B)

    # Assert
    np.testing.assert_array_equal(state_space.A, expected_state_space_a)
    np.testing.assert_array_equal(state_space.B, expected_state_space_b)


def test_state_space_predict():
    """Test the state space class."""
    # Arrange
    exp_predict = np.ones((2, 1))
    state = np.ones((2, 1))

    A = np.eye(2)
    B = np.eye(2)

    # Act
    state_space = StateSpace(A, B)

    predict = state_space.predict(state)

    # Assert
    np.testing.assert_array_equal(predict, exp_predict)


@pytest.mark.parametrize("delta_time", [1.0, 0.1, 0.01])
def test_state_space_cont2disc(delta_time):
    """Test the state space class."""
    # Arrange
    A = np.eye(2)
    B = np.eye(2)
    exp_disc_a = np.exp(delta_time) * A
    exp_disc_b = exp_disc_a @ B * delta_time
    state_space = StateSpace(A, B)

    # Act
    disc_state_space = state_space.cont2disc(dt=delta_time)

    # Assert
    np.testing.assert_array_equal(disc_state_space[0], exp_disc_a)
    np.testing.assert_array_equal(disc_state_space[1], exp_disc_b)
