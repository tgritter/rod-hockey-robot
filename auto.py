"""Run the autonomous puck-control loop.

Usage:
  python auto.py [--player left-defense-hockey-player] [--cycles N]
"""

import argparse
import asyncio

from viam.robot.client import RobotClient

from robot.const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID
from robot.autonomy import run


async def _main():
    parser = argparse.ArgumentParser(description="Autonomous puck control")
    parser.add_argument("--player", default="left-defense-hockey-player")
    parser.add_argument("--cycles", type=int, default=None,
                        help="stop after N cycles (default: run until interrupted)")
    args = parser.parse_args()

    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID,
    )
    machine = await RobotClient.at_address(ROBOT_ADDRESS, opts)
    try:
        await run(machine, args.player, max_cycles=args.cycles)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(_main())
