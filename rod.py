"""Drive and train one rod for puck control.

Usage:
  python rod.py collect [--rod left-defense]
  python rod.py control --target X Y [--rod left-defense]
"""

import argparse
import asyncio

from viam.robot.client import RobotClient

from robot.const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID
from robot.rod_model import dataset_path, load_dataset, RodModel
from robot.rod_session import collect_dataset, carry_puck


async def _connect():
    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID,
    )
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)


async def _main():
    parser = argparse.ArgumentParser(description="One-rod puck control")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Drive the rod and record real samples")
    p_collect.add_argument("--rod", default="left-defense-hockey-player")

    p_control = sub.add_parser("control", help="Carry the puck to a target")
    p_control.add_argument("--rod", default="left-defense-hockey-player")
    p_control.add_argument("--target", nargs=2, type=float, required=True,
                           metavar=("X", "Y"))

    args = parser.parse_args()

    machine = await _connect()
    try:
        if args.cmd == "collect":
            await collect_dataset(machine, args.rod)
        else:
            samples = load_dataset(dataset_path(args.rod))
            print(f"Loaded {len(samples)} samples; fitting model.")
            model = RodModel(samples)
            await carry_puck(machine, args.rod, model, tuple(args.target))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(_main())
