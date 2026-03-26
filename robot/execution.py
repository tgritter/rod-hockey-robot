from viam.robot.client import RobotClient
from viam.components.motor import Motor

from .const import (
    ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID,
    TICKS_PER_ROTATION,
)
from engine.constants import PlayerID


def _robot_credentials(player_id):
    return ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID


async def execute_sequence(sequence, player_id=PlayerID.CENTER):
    """Execute a calibrated instruction sequence, then reset to home position."""
    if not sequence:
        print("Empty sequence.")
        return

    print(f"Executing sequence ({len(sequence)} steps, player={player_id.name})")

    address, api_key, api_key_id = _robot_credentials(player_id)
    opts = RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    robot = await RobotClient.at_address(address, opts)
    part_prefix = player_id.get_prefix()
    motor_move = Motor.from_robot(robot=robot, name=part_prefix + "motor-movement")
    motor_rot  = Motor.from_robot(robot=robot, name=part_prefix + "motor-rotation")

    net_move   = 0.0
    net_rotate = 0.0

    for motor, ticks, rpm in sequence:
        revs = ticks / TICKS_PER_ROTATION
        if motor == "move":
            await motor_move.go_for(rpm=rpm, revolutions=revs)
            net_move += revs
        elif motor == "rotate":
            await motor_rot.go_for(rpm=rpm, revolutions=revs)
            net_rotate += revs

    # Reset to home
    print("Resetting to home...")
    if net_move != 0:
        await motor_move.go_for(rpm=200, revolutions=-net_move)
    if net_rotate != 0:
        await motor_rot.go_for(rpm=200, revolutions=-net_rotate)

    print("Done.")
