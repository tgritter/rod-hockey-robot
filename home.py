"""Home one or more player gantries.

Each gantry must be homed (driven to its limit switch to find zero) once per
viam-server session before plays will run -- otherwise plays fail with
"cannot get position until gantry '...' is homed". Run this after a robot restart.

Usage:
  python3 home.py --player center
  python3 home.py --player center --player left_wing
  python3 home.py --all

Players: center, left_d, right_d, left_wing, right_wing
"""

import argparse
import asyncio

from viam.robot.client import RobotClient
from viam.components.gantry import Gantry

from robot.const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID

# Map each player to its gantry component name (the lateral slide for that rod).
_PLAYER_TO_GANTRY = {
    "center":     "center-gantry",
    "left_wing":  "left-wing-gantry",
    "right_wing": "right-wing-gantry",
    "left_d":     "left-defense-gantry",
    "right_d":    "right-defense-gantry",
}


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID,
    )
    return await RobotClient.at_address(ROBOT_ADDRESS, opts)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", action="append", choices=_PLAYER_TO_GANTRY.keys(),
                        help="player to home; repeat for several")
    parser.add_argument("--all", action="store_true", help="home every player")
    args = parser.parse_args()

    # Home everything with --all, otherwise just the players that were passed.
    players = list(_PLAYER_TO_GANTRY) if args.all else (args.player or [])
    if not players:
        parser.error("pass --player <name> (repeatable) or --all")

    robot = await connect()
    try:
        for p in players:
            name = _PLAYER_TO_GANTRY[p]
            print(f"Homing {name}...")
            # Homing talks to hardware and can fail (offline part, bad limit switch);
            # catch per-rod so one failure doesn't abort the rest of an --all run.
            try:
                ok = await Gantry.from_robot(robot, name).home()
                print(f"  {'OK' if ok else 'FAILED'}: {name}")
            except Exception as e:
                print(f"  ERROR: {name}: {e}")
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
