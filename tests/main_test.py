"""public doc string."""

import numpy as np

from uav_sensor_fusion.definitions import NUM_INPUTS
from uav_sensor_fusion.simulation import run_simulation


def test_run_simulation():
    """Test sample function."""
    # Arrange
    variances = (0.0, 0.0, 0.0, 0.0)
    num_steps = 5

    # Act
    truth, estimate = run_simulation(
        initial_state=(0.0, 0.0),
        control_inputs=np.zeros((NUM_INPUTS, num_steps)),
        variances=variances,
        wind_speed_x=0.0,
        show_simulation=False,
    )

    truth = np.array(truth)
    estimate = np.array(estimate)

    # Assert
    np.testing.assert_almost_equal(truth, estimate, decimal=3)
