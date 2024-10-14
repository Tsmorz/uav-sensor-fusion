"""public doc string."""

from typing import Optional

import numpy as np

from src.math_utils import matrix_exponential


class StateSpace:
    """Class representing a state space model."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ):
        """
        Initialize a state space model.

        :param A: State transition matrix (n x n)
        :param B: Input matrix (n x m)
        :param C: Output matrix (p x n)
        :param D: Feedforward matrix (p x m)
        """
        self.A = A
        self.B = B

        if C is None:
            C = np.eye(A.shape[0])
        if D is None:
            D = np.zeros((C.shape[0], B.shape[1]))
        self.C = C
        self.D = D
        pass

    def predict(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Predict the next state with the given control input.

        :param state: current state
        :param control: control input
         :return: the next state predicted with the given control input
        """
        return self.A @ state + self.B @ control

    def cont2disc(self, dt) -> tuple[np.ndarray, np.ndarray]:
        """
        Discretize the state space model using the given time step.

        :param dt: time step in seconds
        :return: discrete state transition and input matrices
        """
        mat_exp_a = matrix_exponential(self.A, dt)
        A_disc = mat_exp_a
        B_disc = mat_exp_a @ self.B * dt
        return A_disc, B_disc
