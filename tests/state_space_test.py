"""public doc string."""

import numpy as np
import pytest

from uav_sensor_fusion.state_space import StateSpace


def test_state_space():
    """Test the state space class."""
    # Arrange
    expected_state_space_a = np.eye(2)
    expected_state_space_b = np.eye(2)
    expected_state_space_c = np.eye(2)
    expected_state_space_d = np.zeros((2, 2))

    A = np.eye(2)
    B = np.eye(2)

    # Act
    state_space = StateSpace(A=A, B=B, C=None, D=None)

    # Assert
    np.testing.assert_array_equal(state_space.A, expected_state_space_a)
    np.testing.assert_array_equal(state_space.B, expected_state_space_b)
    np.testing.assert_array_equal(state_space.C, expected_state_space_c)
    np.testing.assert_array_equal(state_space.D, expected_state_space_d)


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
