"""Command-line entrypoint for the UAV sensor-fusion simulation."""

import argparse

from uav_sensor_fusion.simulation import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation inputs")
    parser.add_argument("--hide", action="store_true")

    args = parser.parse_args()

    main(show_sim=not args.hide)
