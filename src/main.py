"""public doc string."""

import matplotlib.pyplot as plt
import numpy as np
from ground_model_utils import ground
from loguru import logger
from pressure_utils import PressureSensor

from definitions import EPSILON, FIG_SIZE


def fx(state: np.ndarray, x_old: np.ndarray) -> np.ndarray:
    """Find the state estimate given the state and previous state.

    :param state: current state
    :param x_old: previous state
    :return: the state estimate
    """
    num_states = 2
    x, y = state

    x_old = np.array([[x_old[0]], [x_old[1]]])
    A = np.eye(num_states)
    B = np.eye(num_states)
    est_u = np.linalg.inv(B.T @ B) @ B.T @ (np.array([[x], [y]]) - A @ x_old)

    pressure_sensor = PressureSensor()
    f1 = pressure_sensor.height2pressure(height=y)
    f2 = y - ground(x)
    f3 = est_u[0, 0]
    f4 = est_u[1, 0]

    f = np.array([[f1], [f2], [f3], [f4]])

    return f


def partial_f(state: np.ndarray, x_old: np.ndarray) -> np.ndarray:
    """Find the partial derivatives of the given state.

    :param state: current state
    :param x_old: previous state
    :return: the partial derivatives of the state
    """
    x, y = state[0], state[1]
    dx, dy = EPSILON, EPSILON

    dfdx1 = (fx(np.array([x + dx, y]), x_old) - fx(np.array([x, y]), x_old)) / dx
    dfdx2 = (fx(np.array([x, y + dy]), x_old) - fx(np.array([x, y]), x_old)) / dy

    df = np.hstack((dfdx1, dfdx2))

    return df


def cost(x, y, measurement, var):
    """Create a cost function to minimize the state uncertainty.

    :param x: current distance
    :param y: current height
    :param measurement: measurement
    :param var: covariance matrix
    """
    p, r, x_old, u = measurement

    f = fx(np.array([x, y]), x_old)

    b = np.array([[p], [r], [u[0, 0]], [u[1, 0]]])

    J = f - b

    W = var.T @ var + 0.1 * np.eye(4)
    c = J.T @ np.linalg.inv(W) @ J
    return c


def grad_descent(state, measurement, var):
    """Perform gradient descent on the cost function.

    :param state: current state
    :param measurement: measurement
    :param var: covariance matrix
    """
    p, r, x_old, u = measurement
    x, y = state

    X = np.array([[x], [y]])

    b = np.array([[p], [r], [u[0, 0]], [u[1, 0]]])

    states = [(x, y)]
    for _i in range(100):
        dfdx = partial_f((x, y), x_old)

        f = fx((x, y), x_old)

        W = var.T @ var + 0.1 * np.eye(4)
        invW = np.linalg.inv(W)
        deltaX = np.linalg.inv(dfdx.T @ invW @ dfdx) @ dfdx.T @ (b - f)

        X = X + 5 * deltaX

        x = X[0, 0]
        y = X[1, 0]

        states.append((x, y))

    return states


def cost_contour(x: np.ndarray, y: np.ndarray, m: float, var: float):
    """Visualize the cost function gradient.

    :param x: x coordinate
    :param y: y coordinate
    :param m: measurement from sensor
    :param var: measurement noise
    """
    J = np.zeros((np.shape(x)[0], np.shape(y)[0]))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(y)[0]):
            J[j, i] = cost(x[i], y[j], m, var)
    return J


def prediction(state: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Predict the next state given the state and control input.

    :param state: current state
    :param u: control input
    :return: the next state predicted given the state and control input
    """
    u = np.reshape(u, (2, 1))
    guess = state + u
    return guess


def main():
    """Run the main function."""
    # create environment
    plt.figure(2, figsize=FIG_SIZE)
    x = np.linspace(0, 100, 40)
    y = np.linspace(0, 50, 40)
    [X, Y] = np.meshgrid(x, y)
    g = ground(x)

    # initial state
    state = np.array([[5.0], [10.0]])
    var = np.array([[1.0, 0.1, 0.5, 0.5]])

    # control commands
    t = 40
    u1 = 2 * np.ones(t)
    u2 = np.sin(4 * np.arange(t) / t)
    us = np.vstack((u1, u2))

    # store previous states
    prev = [(state[0, 0], state[1, 0])]
    prev_pred = [(state[0, 0], state[1, 0])]

    wind = +1.0

    # find cost contours every step
    for i in range(len(u1) - 1):
        # predictions and control commands
        guess = prediction(state, us[:, i])
        u = np.reshape(us[:, i], (2, 1))
        state += u + np.random.normal(0, scale=var[0, 3], size=(2, 1))
        state[0, 0] -= np.random.normal(wind, scale=wind / 2)

        # measurements
        pressure_sensor = PressureSensor()
        p = pressure_sensor.height2pressure(height=state[1, 0])
        r = state[1, 0] - ground(state[0, 0])
        m = (
            p + np.random.normal(0, scale=var[0, 0]),
            r + np.random.normal(0, scale=var[0, 1]),
            prev[i],
            u,
        )

        # store ground truth
        prev.append((state[0, 0], state[1, 0]))

        # store prediction
        sol = grad_descent((guess[0, 0] - 10, guess[1, 0] + 10), m, var)
        sx, sy = zip(*sol)
        prev_pred.append((sx[-1], sy[-1]))

        # plot new data
        plt.cla()

        # plot measurements
        pressure_sensor = PressureSensor()
        h = pressure_sensor.pressure2height(pressure=m[0])

        plt.plot([0, np.max(x)], [h, h], "--", color=[0, 1, 1])
        plt.plot(
            [state[0], state[0]], [state[1], state[1] - m[1]], "--", color=[0, 1, 0.5]
        )
        plt.plot(state[0], state[1], "k*")

        # gradient descent
        plt.plot(sx[-1], sy[-1], "y*")
        plt.plot(sx, sy, "r--")

        # ground truth
        prevx, prevy = zip(*prev)
        prevx_pred, prevy_pred = zip(*prev_pred)

        plt.plot(prevx[0] + sum(us[0, 0:i]), prevy[0] + sum(us[1, 0:i]), "ro")
        plt.plot(prevx, prevy, "k--")
        plt.plot(prevx_pred, prevy_pred, "y--")
        plt.legend(
            [
                "pressure measurement",
                "lidar measurement",
                "ground truth",
                "maximum likelihood estimate",
                "prediction without measurements",
            ]
        )

        # calculate cost function contour
        J = cost_contour(x, y, m, var)
        plt.contourf(X, Y, J, 100, cmap="RdBu_r")
        plt.fill_between(x, 0, g, color="green")

        plt.xlabel("x-axis (m)")
        plt.ylabel("y-axis (m)")
        plt.title("Nonlinear Least Squares Drone Localization")
        plt.xlim([0, 100])
        plt.ylim([0, 40])

        plt.show()
        plt.close()

    diffxLS = np.array(prevx) - np.array(prevx_pred)
    diffx = np.array(prevx) - prevx[0] - np.cumsum(us[0, :])

    diffyLS = np.array(prevy) - np.array(prevy_pred)
    diffy = np.array(prevy) - prevy[0] - np.cumsum(us[1, :])

    all = np.vstack((diffxLS, diffx, diffyLS, diffy))
    lim = np.max(abs(all))
    plt.figure(3, figsize=FIG_SIZE)
    plt.plot(diffx, "b-")
    plt.plot(diffxLS, "b--")
    plt.plot(diffy, "r-")
    plt.plot(diffyLS, "r--")
    plt.legend(
        [
            "x-error - w/o measurements",
            "x-error - w/ measurements",
            "y-error - w/o measurements",
            "y-error - w/ measurements",
        ]
    )
    plt.ylim([-lim - 1, lim + 1])
    plt.xlabel("time (s)")
    plt.ylabel("position error (m)")
    plt.grid(True)
    plt.show()

    logger.info(np.std(diffx))
    logger.info(np.std(diffy))

    logger.info(np.std(diffxLS))
    logger.info(np.std(diffyLS))
    return


if __name__ == "__main__":
    main()
