"""public doc string."""

import numpy as np

from definitions import (
    GAS_CONSTANT,
    GRAVITY_ACCELERATION_METRIC,
    MOLAR_MASS,
    PRESSURE_SEA_LEVEL,
    PRESSURE_VARIANCE,
    SEA_LEVEL_METERS,
    TEMP_KELVIN,
)


class PressureSensor:
    """
    Pressure sensor class to find pressure or height based on the other.

    Source - https://en.wikipedia.org/wiki/Barometric_formula
    """

    def __init__(self, noise_variance: float = PRESSURE_VARIANCE):
        """Create a pressure sensor."""
        self.h0 = SEA_LEVEL_METERS
        self.p0 = PRESSURE_SEA_LEVEL
        self.g = GRAVITY_ACCELERATION_METRIC
        self.M = MOLAR_MASS
        self.R = GAS_CONSTANT
        self.T = TEMP_KELVIN
        self.variance = noise_variance
        pass

    def height2pressure(self, height: float) -> float:
        """
        Find the pressure given a height.

        :param self: self
        :param height: the height in meters
        :return: the pressure in Pascals
        """
        noise = np.random.normal(0, scale=self.variance)
        p = self.p0 * np.exp(
            -self.g * self.M * (height - self.h0) / self.R / self.T
        )

        return p + noise

    def pressure2height(self, pressure: float) -> float:
        """
        Find the height given a pressure.

        :param self: self
        :param pressure: the pressure measurement in Pascals
        :return: the height in meters
        """
        noise = np.random.normal(0, scale=self.variance)
        h = (
            self.h0
            - np.log(pressure / self.p0) * self.R * self.T / self.g / self.M
        )

        return h + noise
