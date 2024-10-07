"""public doc string."""

import numpy as np
from loguru import logger

from definitions import EPSILON, NUM_INPUTS, NUM_STATES, WIND_SPEED_X_AXIS
from ground_model_utils import ground
from pressure_utils import PressureSensor
from src.plot_utils import plot_simulation, plot_state_error


def fx(state: np.ndarray, x_old: np.ndarray) -> np.ndarray:
    """
    Find the state estimate given the state and previous state.

    :param state: current state
    :param x_old: previous state
    :return: the state estimate
    """
    x, y = state

    x_old = np.array([[x_old[0]], [x_old[1]]])
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


def partial_f(state: np.ndarray, x_old: np.ndarray) -> np.ndarray:
    """
    Find the partial derivatives of the given state.

    :param state: current state
    :param x_old: previous state
    :return: the partial derivatives of the state
    """
    x, y = state[0], state[1]
    dx, dy = EPSILON, EPSILON

    df_dx1 = (fx(np.array([x + dx, y]), x_old) - fx(np.array([x, y]), x_old)) / dx
    df_dx2 = (fx(np.array([x, y + dy]), x_old) - fx(np.array([x, y]), x_old)) / dy

    df = np.hstack((df_dx1, df_dx2))

    return df


def cost_fxn(x: float, y: float, measurement: tuple, var: np.ndarray) -> float:
    """
    Create a cost function to minimize the state uncertainty.

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
    return float(c[0][0])


def grad_descent(state: tuple, measurement: tuple, var: np.ndarray) -> list:
    """
    Perform gradient descent on the cost function.

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
        df_dx = partial_f(np.array([x, y]), x_old)

        f = fx(np.array([x, y]), x_old)

        W = var.T @ var + 0.1 * np.eye(4)
        invW = np.linalg.inv(W)
        deltaX = np.linalg.inv(df_dx.T @ invW @ df_dx) @ df_dx.T @ (b - f)

        X = X + 5 * deltaX

        x = X[0, 0]
        y = X[1, 0]

        states.append((x, y))

    return states


def cost_contour(x: np.ndarray, y: np.ndarray, m: tuple, var: np.ndarray) -> np.ndarray:
    """
    Visualize the cost function gradient.

    :param x: x coordinate
    :param y: y coordinate
    :param m: measurement from sensor
    :param var: measurement noise
    """
    J = np.zeros((np.shape(x)[0], np.shape(y)[0]))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(y)[0]):
            J[j, i] = cost_fxn(float(x[i]), float(y[j]), m, var)
    return J


def prediction(state: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Predict the next state given the state and control input.

    :param state: current state
    :param u: control input
    :return: the next state predicted given the state and control input
    """
    u = np.reshape(u, (NUM_INPUTS, 1))
    guess = state + u
    return guess


def main() -> None:
    """Run the main function."""
    # create environment
    x = np.linspace(0, 100, 40)
    y = np.linspace(0, 50, 40)

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

    # find cost contours every step
    for i in range(len(u1) - 1):
        # predictions and control commands
        guess = prediction(state, us[:, i])
        u = np.reshape(us[:, i], (2, 1))
        state += u + np.random.normal(0, scale=var[0, 3], size=(2, 1))
        state[0, 0] -= np.random.normal(WIND_SPEED_X_AXIS, scale=WIND_SPEED_X_AXIS / 2)

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

        # plot measurements
        pressure_sensor = PressureSensor()
        h = pressure_sensor.pressure2height(pressure=m[0])

        # ground truth
        prev_x, prev_y = zip(*prev)
        prev_x_pred, prev_y_pred = zip(*prev_pred)

        # calculate cost function contour
        j = cost_contour(x, y, m, var)
        plot_simulation(state, h, j, sx, sy, prev, prev_pred, us, m, i)

    diffxLS = np.array(prev_x) - np.array(prev_x_pred)
    diffx = np.array(prev_x) - prev_x[0] - np.cumsum(us[0, :])

    diffyLS = np.array(prev_y) - np.array(prev_y_pred)
    diffy = np.array(prev_y) - prev_y[0] - np.cumsum(us[1, :])

    plot_state_error(diffxLS, diffx, diffyLS, diffy)

    logger.info(np.std(diffx))
    logger.info(np.std(diffy))
    logger.info(np.std(diffxLS))
    logger.info(np.std(diffyLS))

    return


if __name__ == "__main__":
    main()
