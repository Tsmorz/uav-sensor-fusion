"""Plotting helpers and the live simulation visualizer."""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import AbstractMovieWriter
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm

from uav_sensor_fusion.definitions import (
    FIG_SIZE,
    NUM_COST_CONTOURS,
    NUM_STATES,
    SIMULATION_DIMENSIONS,
)
from uav_sensor_fusion.ground_model_utils import ground
from uav_sensor_fusion.pressure_utils import PressureSensor

# ---------------------------------------------------------------------------
# Palette (validated colorblind-safe categorical hues; warm hues read against
# the sequential-blue cost field). Identity is never carried by color alone:
# every series is direct-labeled in the legend and drawn with a distinct mark.
# ---------------------------------------------------------------------------
GROUND_TRUTH = "#0b0b0b"  # ink
ESTIMATE = "#eb6834"  # orange  - maximum likelihood estimate (w/ measurements)
PREDICTION = "#1baf7a"  # aqua    - dead reckoning (w/o measurements)
PRESSURE = "#eda100"  # yellow  - pressure (barometric) measurement
RANGE = "#e34948"  # red     - range / time-of-flight measurement
TERRAIN = "#c9bfa3"  # muted sand - the known ground, not a data series

# A 2px white ring keeps marks legible over the cost field (mark spec / relief).
_HALO = [pe.withStroke(linewidth=3, foreground="white")]


def plot_state_error(
    diffxLS: np.ndarray,
    diffx: np.ndarray,
    diffyLS: np.ndarray,
    diffy: np.ndarray,
) -> None:
    """Plot the state error after the simulation is complete.

    :param diffxLS: x position error with measurements
    :param diffx: x position error without measurements
    :param diffyLS: y position error with measurements
    :param diffy: y position error without measurements
    :return: None
    """
    all_errors = np.vstack((diffxLS, diffx, diffyLS, diffy))
    lim = float(np.max(abs(all_errors)))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # Color = axis (x vs y); line style = with vs without measurements.
    ax.plot(diffx, color=ESTIMATE, ls="-", lw=2, label="x - w/o measurements")
    ax.plot(diffxLS, color=ESTIMATE, ls="--", lw=2, label="x - w/ measurements")
    ax.plot(diffy, color=PREDICTION, ls="-", lw=2, label="y - w/o measurements")
    ax.plot(diffyLS, color=PREDICTION, ls="--", lw=2, label="y - w/ measurements")

    ax.axhline(0, color="#c3c2b7", lw=1, zorder=0)
    ax.set_ylim(-lim - 1, lim + 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (m)")
    ax.set_title("State estimation error")
    ax.grid(True, color="#e1e0d9", lw=0.8)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    plt.show()


class SimulationVisualizer:
    """Render the localization simulation into a single, reused figure.

    One figure is created up front and redrawn in place every step, so a run
    shows as a live animation instead of one blocking window per step. Pass a
    ``save_path`` (``.gif`` or ``.mp4``) to also record the run to disk.
    """

    def __init__(
        self,
        *,
        live: bool = True,
        save_path: str | None = None,
        fps: int = 10,
    ) -> None:
        """Set up the figure and, optionally, an animation writer.

        :param live: draw each frame to screen as the simulation runs
        :param save_path: write the run to this file (``.gif`` or ``.mp4``)
        :param fps: frames per second for the saved animation
        """
        self.live = live
        self.fig, self.ax = plt.subplots(figsize=FIG_SIZE)
        self._cbar: Colorbar | None = None
        self._writer: AbstractMovieWriter | None = None

        # Cost-field grid, matching cost_contours() exactly so the filled
        # contour lands on the right coordinates.
        gx = np.linspace(0, SIMULATION_DIMENSIONS[0], NUM_COST_CONTOURS)
        gy = np.linspace(0, SIMULATION_DIMENSIONS[1], NUM_COST_CONTOURS)
        self._gx = gx
        self._grid_x, self._grid_y = np.meshgrid(gx, gy)
        self._ground = ground(gx)

        if live:
            plt.ion()
            self.fig.show()

        if save_path is not None:
            self._writer = self._make_writer(save_path, fps)
            self._writer.setup(self.fig, save_path, dpi=100)

    @staticmethod
    def _make_writer(save_path: str, fps: int):
        """Pick a Pillow (gif) or FFMpeg (mp4) writer from the file suffix."""
        from matplotlib.animation import FFMpegWriter, PillowWriter

        if save_path.lower().endswith(".mp4"):
            return FFMpegWriter(fps=fps)
        return PillowWriter(fps=fps)

    def update(
        self,
        state,
        sx,
        sy,
        prev,
        prev_pred,
        controls,
        measurements,
        i,
        variances_array,
    ) -> None:
        """Redraw the figure for the current simulation step.

        :param state: current ground-truth state
        :param sx: x-values along the gradient-descent path
        :param sy: y-values along the gradient-descent path
        :param prev: ground-truth state history
        :param prev_pred: maximum-likelihood estimate history
        :param controls: control-input history
        :param measurements: current measurement tuple
        :param i: current step index
        :param variances_array: measurement noise variances (1 x n)
        """
        ax = self.ax
        ax.clear()

        # --- cost field (sequential magnitude) -----------------------------
        cost = cost_contours(measurement=measurements, variances=variances_array)
        floor = max(float(cost[cost > 0].min()) if np.any(cost > 0) else 1e-6, 1e-6)
        norm = LogNorm(vmin=floor, vmax=float(cost.max()))
        field = ax.contourf(
            self._grid_x,
            self._grid_y,
            np.clip(cost, floor, None),
            levels=30,
            cmap="Blues",
            norm=norm,
            zorder=0,
        )
        if self._cbar is None:
            self._cbar = self.fig.colorbar(field, ax=ax, pad=0.02)
            self._cbar.set_label("localization cost (log scale)")
        else:
            self._cbar.update_normal(field)

        # --- known terrain -------------------------------------------------
        ax.fill_between(
            self._gx, 0, self._ground, color=TERRAIN, zorder=1, label="terrain"
        )

        # --- measurements (auxiliary guide lines) --------------------------
        pressure_variance = variances_array[0, 1]
        h = PressureSensor(noise_variance=pressure_variance).pressure2height(
            pressure=measurements[0]
        )
        ax.axhline(h, ls="--", lw=1.5, color=PRESSURE, zorder=2, label="pressure meas.")
        ax.plot(
            [state[0], state[0]],
            [state[1], state[1] - measurements[1]],
            ls="--",
            lw=1.5,
            color=RANGE,
            zorder=2,
            label="range meas.",
        )

        # --- prediction without measurements (dead reckoning) --------------
        prev_x, prev_y = zip(*prev, strict=False)
        prev_x_pred, prev_y_pred = zip(*prev_pred, strict=False)
        ax.plot(
            prev_x[0] + sum(controls[0, 0:i]),
            prev_y[0] + sum(controls[1, 0:i]),
            "o",
            ms=9,
            color=PREDICTION,
            path_effects=_HALO,
            zorder=4,
            label="prediction (w/o meas.)",
        )

        # --- maximum likelihood estimate (with measurements) ---------------
        ax.plot(sx, sy, ":", lw=1.5, color=ESTIMATE, zorder=3)  # descent path
        ax.plot(
            prev_x_pred,
            prev_y_pred,
            "--",
            lw=2,
            color=ESTIMATE,
            path_effects=_HALO,
            zorder=4,
            label="estimate (w/ meas.)",
        )
        ax.plot(
            sx[-1],
            sy[-1],
            "*",
            ms=15,
            color=ESTIMATE,
            path_effects=_HALO,
            zorder=6,
        )

        # --- ground truth --------------------------------------------------
        ax.plot(
            prev_x,
            prev_y,
            "--",
            lw=2,
            color=GROUND_TRUTH,
            path_effects=_HALO,
            zorder=5,
            label="ground truth",
        )
        ax.plot(
            state[0],
            state[1],
            "*",
            ms=15,
            color=GROUND_TRUTH,
            path_effects=_HALO,
            zorder=7,
        )

        # --- chrome --------------------------------------------------------
        ax.set_xlabel("x-axis (m)")
        ax.set_ylabel("y-axis (m)")
        ax.set_title("Nonlinear least squares drone localization")
        ax.set_xlim(0, SIMULATION_DIMENSIONS[0])
        ax.set_ylim(0, SIMULATION_DIMENSIONS[1])
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

        self._draw()

    def _draw(self) -> None:
        """Flush the current frame to the screen and/or the saved animation."""
        if self._writer is not None:
            self._writer.grab_frame()
        if self.live:
            # Headless backends (e.g. Agg during --save) have no event loop.
            try:
                self.fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:  # noqa: S110
                pass

    def close(self) -> None:
        """Finish any saved animation and release the figure."""
        if self._writer is not None:
            self._writer.finish()
            self._writer = None
        if self.live:
            plt.ioff()
        plt.close(self.fig)


def cost_fxn(x: float, y: float, measurement: tuple, var: np.ndarray) -> float:
    """Create a cost function to minimize the state uncertainty.

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
    """Find the state estimate given the state and previous state.

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
    """Visualize the cost function gradient.

    :param measurement: measurement from sensor
    :param variances: measurement noise
    :return: a 2D array representing the cost function at each point
    """
    x = np.linspace(0, SIMULATION_DIMENSIONS[0], NUM_COST_CONTOURS)
    y = np.linspace(0, SIMULATION_DIMENSIONS[1], NUM_COST_CONTOURS)

    cost = np.zeros((np.shape(x)[0], np.shape(y)[0]))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(y)[0]):
            cost[j, i] = cost_fxn(float(x[i]), float(y[j]), measurement, variances)
    return cost
