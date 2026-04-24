# Connect to the machine and print every resource name it exposes.
# Use when motor/camera names in code drift from the Viam config.

import asyncio
from viam.robot.client import RobotClient
from robot.const import ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID


async def main():
    # Auth via API key pulled from .env by robot/const.py
    opts = RobotClient.Options.with_api_key(api_key=ROBOT_API_KEY, api_key_id=ROBOT_API_KEY_ID)
    robot = await RobotClient.at_address(ROBOT_ADDRESS, opts)

    # Sort by subtype (motor/camera/board/...) then name — motors group together
    print(f"Resources on {ROBOT_ADDRESS}:\n")
    for r in sorted(robot.resource_names, key=lambda x: (x.subtype, x.name)):
        print(f"  [{r.subtype}] {r.name}")

    await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
