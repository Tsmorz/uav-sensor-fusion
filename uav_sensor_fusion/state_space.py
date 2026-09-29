"""public doc string."""

import numpy as np

from uav_sensor_fusion.math_utils import matrix_exponential


class StateSpace:
    """Class representing a state space model."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray | None = None,
        D: np.ndarray | None = None,
    ):
        """Initialize a state space model.

        :param A: State transition matrix (n x n)
        :param B: Input matrix (n x m)
        :param C: Output matrix (p x n)
        :param D: Feedforward matrix (p x m)
        """
        if C is None:
            C = np.eye(A.shape[0])  # full state feedback
        if D is None:
            D = np.zeros((A.shape[0], B.shape[1]))

        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def predict(
        self, state: np.ndarray, control: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict the next state with the given control input.

        :param state: current state
        :param control: control input
         :return: the next state predicted with the given control input
        """
        if control is None:
            control = np.zeros((np.shape(self.B)[1], 1))

        return self.A @ state + self.B @ control

    def cont2disc(self, dt) -> tuple[np.ndarray, np.ndarray]:
        """Discretize the state space model using the given time step.

        :param dt: time step in seconds
        :return: discrete state transition and input matrices
        """
        mat_exp_a = matrix_exponential(self.A, dt)
        A_disc = mat_exp_a
        B_disc = mat_exp_a @ self.B * dt
        return A_disc, B_disc
