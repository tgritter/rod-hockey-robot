"""Run the coordinated five-rod relay routine.

Usage:
  python3 routine.py            # run the relay on the robot
  python3 routine.py --dry-run  # print the relay plan, no hardware
"""

import argparse
import asyncio

from viam.robot.client import RobotClient

from robot.const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID
from robot.routine import format_relay_plan, run_relay


async def _main():
    parser = argparse.ArgumentParser(description="Five-rod relay routine")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the relay plan without connecting to hardware")
    args = parser.parse_args()

    if args.dry_run:
        print(format_relay_plan())
        return

    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID,
    )
    machine = await RobotClient.at_address(ROBOT_ADDRESS, opts)
    try:
        await run_relay(machine)
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(_main())
