"""public doc string."""

import matplotlib.pyplot as plt
import numpy as np

from definitions import (
    FIG_SIZE,
    NUM_COST_CONTOURS,
    NUM_STATES,
    SIMULATION_DIMENSIONS,
)
from src.ground_model_utils import ground
from src.pressure_utils import PressureSensor


def plot_state_error(
    diffxLS: np.ndarray,
    diffx: np.ndarray,
    diffyLS: np.ndarray,
    diffy: np.ndarray,
) -> None:
    """
    Plot the state error after the simulation is complete.

    :param diffxLS: x position error without measurements
    :param diffx: x position error with measurements
    :param diffyLS: y position error without measurements
    :param diffy: y position error with measurements
    :return: None
    """
    all_errors = np.vstack((diffxLS, diffx, diffyLS, diffy))

    lim = float(np.max(abs(all_errors)))
    plt.figure(3, figsize=FIG_SIZE)
    plt.plot(diffx, "b-", label="x-error - w/o measurements")
    plt.plot(diffxLS, "b--", label="x-error - w/ measurements")
    plt.plot(diffy, "r-", label="y-error - w/o measurements")
    plt.plot(diffyLS, "r--", label="y-error - w/ measurements")
    plt.legend()

    plt.ylim((-lim - 1, lim + 1))
    plt.xlabel("time (s)")
    plt.ylabel("position error (m)")
    plt.grid(True)
    plt.show()

    return


def plot_simulation(
    state, sx, sy, prev, prev_pred, controls, measurements, i, variances_array
) -> None:
    """
    Plot the simulation visualization after each step.

    :param state: current state
    :param sx: x-axis values for gradient descent
    :param sy: y-axis values for gradient descent
    :param prev: previous state history
    :param prev_pred: previous state prediction history
    :param controls: control inputs history
    :param measurements: measurement
    :param variances_array: noise variances in an array (n x 1)
    :param i: current iteration index
    """
    plt.figure(2, figsize=FIG_SIZE)
    x = np.linspace(0, 100, 40)
    y = np.linspace(0, 50, 40)
    [X, Y] = np.meshgrid(x, y)
    g = ground(x)

    pressure_variance = variances_array[0, 1]
    h = PressureSensor(noise_variance=pressure_variance).pressure2height(
        pressure=measurements[0]
    )

    plt.plot([0, np.max(x)], [h, h], "--", color=[0, 1, 1])
    plt.plot(
        [state[0], state[0]],
        [state[1], state[1] - measurements[1]],
        "--",
        color=[0, 1, 0.5],
    )
    plt.plot(state[0], state[1], "k*")

    # gradient descent
    plt.plot(sx[-1], sy[-1], "y*")
    plt.plot(sx, sy, "r--")

    # ground truth
    prev_x, prev_y = zip(*prev)
    prev_x_pred, prev_y_pred = zip(*prev_pred)

    plt.plot(
        prev_x[0] + sum(controls[0, 0:i]),
        prev_y[0] + sum(controls[1, 0:i]),
        "ro",
    )
    plt.plot(prev_x, prev_y, "k--")
    plt.plot(prev_x_pred, prev_y_pred, "y--")
    plt.legend(
        [
            "pressure measurement",
            "lidar measurement",
            "ground truth",
            "maximum likelihood estimate",
            "prediction without measurements",
        ],
    )

    # calculate cost function contour
    j = cost_contours(measurement=measurements, variances=variances_array)
    plt.contourf(X, Y, j, 100, cmap="RdBu_r")
    plt.fill_between(x, 0, g, color="green")

    plt.xlabel("x-axis (m)")
    plt.ylabel("y-axis (m)")
    plt.title("Nonlinear Least Squares Drone Localization")
    plt.xlim([0, 100])
    plt.ylim([0, 40])

    plt.show()
    plt.close()

    return


def cost_fxn(x: float, y: float, measurement: tuple, var: np.ndarray) -> float:
    """
    Create a cost function to minimize the state uncertainty.

    :param x: current distance
    :param y: current height
    :param measurement: measurement
    :param var: covariance matrix
    """
    epsilon = 1e-1
    p, r, x_old, u = measurement

    f = fx(np.array([x, y]), x_old)

    b = np.array([[p], [r], [u[0, 0]], [u[1, 0]]])

    J = f - b

    W = var.T @ var + epsilon * np.eye(4)
    c = J.T @ np.linalg.inv(W) @ J
    return float(c[0][0])


def fx(state: np.ndarray, x_old: np.ndarray) -> np.ndarray:
    """
    Find the state estimate given the state and previous state.

    :param state: current state
    :param x_old: previous state
    :return: the state estimate
    """
    x, y = state

    x_old = np.reshape(x_old, (NUM_STATES, 1))
    A = np.eye(NUM_STATES)
    B = np.eye(NUM_STATES)
    est_u = np.linalg.inv(B.T @ B) @ B.T @ (np.array([[x], [y]]) - A @ x_old)

    pressure_sensor = PressureSensor()
    f1 = pressure_sensor.height2pressure(height=y)
    f2 = y - ground(x)
    f3 = est_u[0, 0]
    f4 = est_u[1, 0]

    f = np.array([[f1], [f2], [f3], [f4]])

    return f


def cost_contours(measurement: tuple, variances: np.ndarray) -> np.ndarray:
    """
    Visualize the cost function gradient.

    :param measurement: measurement from sensor
    :param variances: measurement noise
    :return: a 2D array representing the cost function at each point
    """
    x = np.linspace(0, SIMULATION_DIMENSIONS[0], NUM_COST_CONTOURS)
    y = np.linspace(0, SIMULATION_DIMENSIONS[1], NUM_COST_CONTOURS)

    cost = np.zeros((np.shape(x)[0], np.shape(y)[0]))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(y)[0]):
            cost[j, i] = cost_fxn(
                float(x[i]), float(y[j]), measurement, variances
            )
    return cost
