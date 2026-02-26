from viam.robot.client import RobotClient
from viam.components.motor import Motor

from .const import (
    ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID,
    MOTOR2_REVS_MIN, MOTOR2_REVS_MAX, MOTOR2_RPM_MAX,
    MOTOR1_REVS_SCALE, MOTOR1_RPM_MAX,
    RW_SLIDE_REVS_MIN, RW_SLIDE_REVS_MAX,
)
from engine.constants import PlayerID


def _scale_action_center(action):
    """Center: motor-movement = lateral slide, motor-rotation = spin."""
    move_revs = -(MOTOR2_REVS_MIN + action[0] * (MOTOR2_REVS_MAX - MOTOR2_REVS_MIN))
    move_rpm  = MOTOR2_RPM_MAX
    rot_revs  = action[1] * MOTOR1_REVS_SCALE
    rot_rpm   = action[2] * MOTOR1_RPM_MAX
    return move_revs, move_rpm, rot_revs, rot_rpm


def _scale_action_right_wing(action):
    """Right wing: motor-movement = lateral slide, motor-rotation = spin."""
    move_revs = -(RW_SLIDE_REVS_MIN + action[0] * (RW_SLIDE_REVS_MAX - RW_SLIDE_REVS_MIN))
    move_rpm  = MOTOR2_RPM_MAX
    rot_revs  = action[1] * MOTOR1_REVS_SCALE
    rot_rpm   = action[2] * MOTOR1_RPM_MAX
    return move_revs, move_rpm, rot_revs, rot_rpm


def _reverse_action(action):
    """Flip rotation direction to produce the reset (return-to-home) move."""
    return [action[0], -action[1], action[2]]


def _robot_credentials(player_id):
    return ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID


async def execute_best_action(action, player_id=PlayerID.CENTER):
    """Connect to the robot for the given player, execute the action, then reverse it to reset position."""
    if not action or len(action) != 3:
        print("Invalid action vector.")
        return

    print(f"Executing action: {action} (player={player_id.name})")

    address, api_key, api_key_id = _robot_credentials(player_id)

    # Connect to robot
    opts = RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    robot = await RobotClient.at_address(address, opts)
    part_prefix = player_id.get_prefix()
    motor_move = Motor.from_robot(robot=robot, name= part_prefix + "motor-movement")
    motor_rot  = Motor.from_robot(robot=robot, name= part_prefix + "motor-rotation")

    scale = _scale_action_right_wing if player_id == PlayerID.RIGHT_WING else _scale_action_center

    # Execute: lateral first, then spin
    move_revs, move_rpm, rot_revs, rot_rpm = scale(action)
    print("Executing...")
    await motor_move.go_for(rpm=move_rpm, revolutions=move_revs)
    await motor_rot.go_for(rpm=rot_rpm, revolutions=rot_revs)

    # Reset to original position
    move_revs_r, move_rpm_r, rot_revs_r, rot_rpm_r = scale(_reverse_action(action))
    print("Resetting position...")
    await motor_move.go_for(rpm=move_rpm_r, revolutions=-move_revs_r)
    await motor_rot.go_for(rpm=rot_rpm_r, revolutions=-rot_revs_r)

    print("Done.")


# --- Standalone test ---
if __name__ == '__main__':
    import asyncio
    import sys

    if len(sys.argv) not in (4, 5):
        print("Usage: python -m robot.execution <distance> <rotation> <rotation_speed> [player]")
        print("  player: center (default) or right_wing")
        sys.exit(1)

    action = [float(v) for v in sys.argv[1:4]]
    player = PlayerID.RIGHT_WING if len(sys.argv) == 5 and sys.argv[4] == 'right_wing' else PlayerID.CENTER
    asyncio.run(execute_best_action(action, player))
