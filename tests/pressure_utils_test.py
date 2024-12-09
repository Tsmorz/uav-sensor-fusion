"""public doc string."""

from src.definitions import PRESSURE_SEA_LEVEL, SEA_LEVEL_METERS
from src.pressure_utils import PressureSensor


def test_pressure_sensor_h2p():
    """Test that the pressure and heights align."""
    sensor = PressureSensor(noise_variance=0.0)
    exp_pressure = PRESSURE_SEA_LEVEL

    pressure = sensor.height2pressure(SEA_LEVEL_METERS)

    assert pressure == exp_pressure


def test_pressure_sensor_p2h():
    """Test that the pressure and heights align."""
    sensor = PressureSensor(noise_variance=0.0)
    exp_height = SEA_LEVEL_METERS

    height = sensor.pressure2height(PRESSURE_SEA_LEVEL)

    assert height == exp_height
