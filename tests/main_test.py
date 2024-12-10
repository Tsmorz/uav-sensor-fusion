"""public doc string."""

import numpy as np

from definitions import NUM_INPUTS
from src.main import run_simulation


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
        show_simulation=False,
    )

    truth = np.array(truth)
    estimate = np.array(estimate)

    # Assert
    np.testing.assert_almost_equal(truth, estimate, decimal=3)
