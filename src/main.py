"""public doc string."""

import argparse
import copy

import numpy as np
from loguru import logger

from definitions import (
    DAMPING_FACTOR,
    DEFAULT_VARIANCES,
    EPSILON,
    LEARNING_RATE,
    NUM_COST_CONTOURS,
    NUM_INPUTS,
    NUM_STATES,
    SIMULATION_DIMENSIONS,
    WIND_SPEED_X_AXIS,
)
from src.ground_model_utils import ground
from src.plot_utils import plot_simulation, plot_state_error
from src.pressure_utils import PressureSensor


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


def partial_f(state: np.ndarray, x_old: np.ndarray) -> np.ndarray:
    """
    Find the partial derivatives of the given state.

    :param state: current state
    :param x_old: previous state
    :return: the partial derivatives of the state
    """
    x, y = state[0], state[1]
    dx, dy = EPSILON, EPSILON

    df_dx1 = (
        fx(np.array([x + dx, y]), x_old) - fx(np.array([x - dx, y]), x_old)
    ) / (2 * dx)
    df_dx2 = (
        fx(np.array([x, y + dy]), x_old) - fx(np.array([x, y - dy]), x_old)
    ) / (2 * dy)

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
    epsilon = 1e-1
    p, r, x_old, u = measurement

    f = fx(np.array([x, y]), x_old)

    b = np.array([[p], [r], [u[0, 0]], [u[1, 0]]])

    J = f - b

    W = var.T @ var + epsilon * np.eye(4)
    c = J.T @ np.linalg.inv(W) @ J
    return float(c[0][0])


def grad_descent(
    state: tuple, measurement: tuple, variances: np.ndarray
) -> list:
    """
    Perform gradient descent on the cost function.

    :param state: current state
    :param measurement: measurement
    :param variances: vector of measurement and state variances
    :return list of estimated states
    """
    pressure, time_of_flight, state_old, control = measurement
    x, y = state
    cov_var = variances.T @ variances

    X = np.array([[x], [y]])

    b = np.array(
        [[pressure], [time_of_flight], [control[0, 0]], [control[1, 0]]]
    )

    states = [(x, y)]
    num_steps = 10000
    for _i in range(num_steps):
        df_dx = partial_f(np.array([x, y]), state_old)

        f = fx(np.array([x, y]), state_old)

        W = cov_var + DAMPING_FACTOR * np.eye(np.shape(cov_var)[1])
        invW = np.linalg.inv(W)
        deltaX = np.linalg.inv(df_dx.T @ invW @ df_dx) @ df_dx.T @ (b - f)

        X = X + LEARNING_RATE * deltaX

        x = X[0, 0]
        y = X[1, 0]

        states.append((x, y))

    return states


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


def run_simulation(
    initial_state: tuple,
    control_inputs: np.ndarray,
    variances: tuple = DEFAULT_VARIANCES,
    show_simulation: bool = True,
) -> tuple[list, list]:
    """
    Run the simulation for a given initial state and all control inputs.

    :param initial_state: initial state
    :param control_inputs: control inputs for all time steps
    :param variances: vector of measurement and state variances
    :param show_simulation: whether to plot the simulation
    :return: list of ground truths and list of estimated states
    """
    # create environment
    pressure_variance, time_of_flight_variance, control_variance, _ = variances

    max_time_steps = np.shape(control_inputs)[1]
    state = np.array([[initial_state[0]], [initial_state[1]]])
    prev = [(state[0, 0], state[1, 0])]
    prev_pred = copy.deepcopy(prev)

    num_inputs = np.shape(control_inputs)[0]

    variances_array = np.array(
        [
            [
                pressure_variance,
                time_of_flight_variance,
                control_variance,
                control_variance,
            ]
        ]
    )

    pressure_sensor = PressureSensor(noise_variance=pressure_variance)

    # find cost contours every step
    for i in range(max_time_steps - 1):
        # predictions and control commands
        guess = prediction(state, control_inputs[:, i])
        u = np.reshape(control_inputs[:, i], (num_inputs, 1))
        state += u + np.random.normal(
            0, scale=control_variance, size=(num_inputs, 1)
        )
        state[0, 0] += WIND_SPEED_X_AXIS  # + np.random.normal(WIND_SPEED_VAR)

        # measurements
        pressure = pressure_sensor.height2pressure(height=float(state[1, 0]))
        time_of_flight = state[1, 0] - ground(state[0, 0])
        measurements = (
            pressure + np.random.normal(0, scale=pressure_variance),
            time_of_flight + np.random.normal(0, scale=time_of_flight_variance),
            prev[i],
            u,
        )

        # store ground truth
        prev.append((state[0, 0], state[1, 0]))

        # store prediction
        offset = 10.0 if show_simulation else 0.0

        sol = grad_descent(
            (guess[0, 0] - offset, guess[1, 0] + offset),
            measurements,
            variances_array,
        )
        sx, sy = zip(*sol)
        prev_pred.append((sx[-1], sy[-1]))

        # plot measurements
        h = pressure_sensor.pressure2height(pressure=measurements[0])

        # calculate cost function contour
        if show_simulation:
            j = cost_contours(
                measurement=measurements, variances=variances_array
            )
            plot_simulation(
                state,
                h,
                j,
                sx,
                sy,
                prev,
                prev_pred,
                control_inputs,
                measurements,
                i,
            )
    return prev, prev_pred


def main(show_sim: bool) -> None:
    """
    Run the main function.

    :param show_sim: whether to show the simulation
    """
    # initial state
    init_x, init_y = 5.0, 10.0

    # control commands
    max_time_steps = 40
    controls_xy = np.vstack(
        (
            2 * np.ones(max_time_steps),  # x input
            np.sin(4 * np.arange(max_time_steps) / max_time_steps),  # y input
        )
    )

    prev, prev_pred = run_simulation(
        initial_state=(init_x, init_y),
        control_inputs=controls_xy,
        show_simulation=show_sim,
    )

    # ground truth
    prev_x, prev_y = zip(*prev)
    prev_x_pred, prev_y_pred = zip(*prev_pred)

    diffxLS = np.array(prev_x) - np.array(prev_x_pred)
    diffx = np.array(prev_x) - prev_x[0] - np.cumsum(controls_xy[0, :])

    diffyLS = np.array(prev_y) - np.array(prev_y_pred)
    diffy = np.array(prev_y) - prev_y[0] - np.cumsum(controls_xy[1, :])

    plot_state_error(diffxLS, diffx, diffyLS, diffy)

    logger.info(
        f"State error w/o measurements:\n"
        f"\t X: {np.std(diffx):.2f}\n"
        f"\t Y: {np.std(diffy):.2f}"
    )
    logger.info(
        f"State error w/ measurements:\n"
        f"\t X: {np.std(diffxLS):.2f}\n"
        f"\t Y: {np.std(diffyLS):.2f}"
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation inputs")
    parser.add_argument("--hide", action="store_true")

    args = parser.parse_args()

    main(show_sim=not args.hide)
