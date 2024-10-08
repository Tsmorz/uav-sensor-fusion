"""public doc string."""

from typing import Any

import numpy as np


def ground(x: np.ndarray[Any, np.dtype]) -> float:
    """
    Parametric function that models the height given a position.

    :param x: the position in meters
    :return: the height in meters
    """
    y = 0.5 * np.cos(x) + 4 * np.cos(0.2 * x) + 0.51 * x - 0.005 * x**2
    return y
