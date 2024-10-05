import numpy as np
import matplotlib.pyplot as plt


def ground(x):
    y = 0.5 * np.cos(x) + 4 * np.cos(0.2 * x) + 0.51 * x - 0.005 * x**2
    # y = 0.15*np.cos(x) + 1
    # y = 0.15*x
    return y


def height2pressure(h, var=0):
    h0 = 0
    p0 = 1013.25
    g = 9.80665
    M = 0.0289644
    R = 8.31432
    T = 273

    noise = np.random.normal(0, scale=var)
    p = p0 * np.exp(-g * M * (h - h0) / R / T)

    return p + noise


def pressure2height(p, var=0):
    h0 = 0
    p0 = 1013.25
    g = 9.80665
    M = 0.0289644
    R = 8.31432
    T = 273

    noise = np.random.normal(0, scale=var)
    h = h0 - np.log(p / p0) * R * T / g / M

    return h + noise


def fx(state, x_old):
    x, y = state

    x_old = np.array([[x_old[0]], [x_old[1]]])
    A = np.eye(2)
    B = np.eye(2)
    est_u = np.linalg.inv(B.T @ B) @ B.T @ (np.array([[x], [y]]) - A @ x_old)

    f1 = height2pressure(y)
    f2 = y - ground(x)
    f3 = est_u[0, 0]
    f4 = est_u[1, 0]

    f = np.array([[f1], [f2], [f3], [f4]])

    return f


def partialf(state, x_old):
    x, y = state
    dx = 0.000001
    dy = 0.000001

    dfdx1 = (fx((x + dx, y), x_old) - fx((x, y), x_old)) / dx
    dfdx2 = (fx((x, y + dy), x_old) - fx((x, y), x_old)) / dy

    df = np.hstack((dfdx1, dfdx2))

    return df


def cost(x, y, measurement, var):
    p, r, x_old, u = measurement

    f = fx((x, y), x_old)

    b = np.array([[p], [r], [u[0, 0]], [u[1, 0]]])

    J = f - b

    W = var.T @ var + 0.1 * np.eye(4)
    c = J.T @ np.linalg.inv(W) @ J
    return c


def gradDescent(state, measurement, var):
    p, r, x_old, u = measurement
    x, y = state

    X = np.array([[x], [y]])

    b = np.array([[p], [r], [u[0, 0]], [u[1, 0]]])

    states = [(x, y)]
    for i in range(100):
        dfdx = partialf((x, y), x_old)

        f = fx((x, y), x_old)

        W = var.T @ var + 0.1 * np.eye(4)
        invW = np.linalg.inv(W)
        deltaX = np.linalg.inv(dfdx.T @ invW @ dfdx) @ dfdx.T @ (b - f)

        X = X + 5 * deltaX

        x = X[0, 0]
        y = X[1, 0]

        states.append((x, y))

    return states


def costContour(x, y, m, var):
    J = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            J[j, i] = cost(x[i], y[j], m, var)
    return J


def prediction(state, u):
    u = np.reshape(u, (2, 1))
    guess = state + u
    return guess


def main():
    # create environment
    plt.figure(2, figsize=(10, 5))
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
        p = height2pressure(state[1, 0])
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
        sol = gradDescent((guess[0, 0] - 10, guess[1, 0] + 10), m, var)
        sx, sy = zip(*sol)
        prev_pred.append((sx[-1], sy[-1]))

        # plot new data
        plt.cla()

        # plot measurements
        h = pressure2height(p=m[0], var=0)
        plt.plot([0, np.max(x)], [h, h], "--", color=[0, 1, 1])
        plt.plot([state[0], state[0]], [state[1], state[1] - m[1]], "--",
                 color=[0, 1, 0.5])
        plt.plot(state[0], state[1], "k*")

        # gradient descent
        plt.plot(sx[-1], sy[-1], "y*")
        # plt.plot(sx,sy,'r--')

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
        J = costContour(x, y, m, var)
        # minr,minc = np.where(J == np.amin(J))
        # plt.plot(x[minc],y[minr],'g*')
        plt.contourf(X, Y, J, 100, cmap="RdBu_r")
        plt.fill_between(x, 0, g, color="green")

        plt.xlabel("x-axis (m)")
        plt.ylabel("y-axis (m)")
        plt.title("Nonlinear Least Squares Drone Localization")
        # plt.axis('equal')
        plt.xlim([0, 100])
        plt.ylim([0, 40])

        plt.show()
        plt.pause(0.01)
    [21]
    diffxLS = np.array(prevx) - np.array(prevx_pred)
    diffx = np.array(prevx) - prevx[0] - np.cumsum(us[0, :])

    diffyLS = np.array(prevy) - np.array(prevy_pred)
    diffy = np.array(prevy) - prevy[0] - np.cumsum(us[1, :])

    all = np.vstack((diffxLS, diffx, diffyLS, diffy))
    lim = np.max(abs(all))
    plt.figure(3, figsize=(8, 4))
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
    plt.grid("on")
    plt.show()

    print(np.std(diffx))
    print(np.std(diffy))

    print(np.std(diffxLS))
    print(np.std(diffyLS))
    return


if __name__ == "__main__":
    main()
