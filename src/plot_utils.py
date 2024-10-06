"""public doc string."""

import matplotlib.pyplot as plt
import numpy as np

from definitions import FIG_SIZE
from src.ground_model_utils import ground


def plot_state_error(
    diffxLS: np.ndarray, diffx: np.ndarray, diffyLS: np.ndarray, diffy: np.ndarray
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


def plot_simulation(state, h, j, sx, sy, prev, prev_pred, us, m, i) -> None:
    """
    Plot the simulation visualization after each step.

    :param state: current state
    :param h: current height
    :param j: cost function result
    :param sx: x-axis values for gradient descent
    :param sy: y-axis values for gradient descent
    :param prev: previous state history
    :param prev_pred: previous state prediction history
    :param us: control inputs history
    :param m: measurement
    :param i: current iteration index
    """
    plt.figure(2, figsize=FIG_SIZE)
    x = np.linspace(0, 100, 40)
    y = np.linspace(0, 50, 40)
    [X, Y] = np.meshgrid(x, y)
    g = ground(x)

    plt.plot([0, np.max(x)], [h, h], "--", color=[0, 1, 1])
    plt.plot(
        [state[0], state[0]],
        [state[1], state[1] - m[1]],
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

    plt.plot(prev_x[0] + sum(us[0, 0:i]), prev_y[0] + sum(us[1, 0:i]), "ro")
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
