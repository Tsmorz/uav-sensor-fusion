# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A planar (2D) drone simulator for state estimation in GPS-denied environments with a known map. The drone localizes itself by solving a nonlinear least-squares problem with the Gauss-Newton algorithm, fusing a barometric pressure measurement and a height-above-ground (time-of-flight) measurement against a known ground map.

The package name is `uav_sensor_fusion`. Model assumptions: a two-dimensional world, a known map, Gaussian process and measurement noise, a pressure measurement, and a height-above-ground measurement.

## Commands

Uses uv + a Taskfile (go-task). Python 3.12. Runtime deps in `[project.dependencies]`; dev tooling (pytest, ruff, mypy, pre-commit) in `[dependency-groups.dev]`.

```bash
task init                 # uv sync + install pre-commit hooks
task run                  # run the simulation with the plots shown
task run -- --hide        # run without showing the simulation plots
task format               # ruff format + ruff check --fix + mypy
task test                 # pytest with coverage over uav_sensor_fusion/
task ci                   # format + test (local CI mirror)
task clean                # remove .venv, caches, build artifacts
```

Run a single test:
```bash
uv run pytest tests/pressure_utils_test.py -v
```

## Layout

- `uav_sensor_fusion/` — the package: simulation logic, sensor models, and plotting.
  - `simulation.py` — the estimation loop (`run_simulation`, `grad_descent`, `prediction`) and `main`.
  - `definitions.py` — physical constants and default variances.
  - `pressure_utils.py`, `ground_model_utils.py`, `math_utils.py`, `state_space.py`, `plot_utils.py` — sensor models and helpers.
- `scripts/simulate.py` — thin CLI entrypoint that calls `simulation.main`.
- `tests/` — pytest suite mirroring the package modules.
