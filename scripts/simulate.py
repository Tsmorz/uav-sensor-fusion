"""Command-line entrypoint for the UAV sensor-fusion simulation."""

import argparse

from uav_sensor_fusion.simulation import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation inputs")
    parser.add_argument(
        "--hide", action="store_true", help="do not display the live animation"
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="record the run to a .gif or .mp4 file",
    )

    args = parser.parse_args()

    main(show_sim=not args.hide, save_path=args.save)
