from viam.robot.client import RobotClient
from viam.components.motor import Motor

from .const import (
    EXEC_ROBOT_ADDRESS, EXEC_API_KEY, EXEC_API_KEY_ID,
    MOTOR2_REVS_MIN, MOTOR2_REVS_MAX, MOTOR2_RPM_MAX,
    MOTOR1_REVS_SCALE, MOTOR1_RPM_MAX,
)


def _scale_action(action):
    """Convert a normalized action vector [0..1] to physical motor parameters."""
    m2_revs = -(MOTOR2_REVS_MIN + action[0] * (MOTOR2_REVS_MAX - MOTOR2_REVS_MIN))
    m2_rpm  = action[1] * MOTOR2_RPM_MAX
    m1_revs = action[2] * -1 * MOTOR1_REVS_SCALE
    m1_rpm  = action[3] * MOTOR1_RPM_MAX
    return m2_revs, m2_rpm, m1_revs, m1_rpm


def _reverse_action(action):
    """Flip motor 1 direction to produce the reset (return-to-home) move."""
    return [action[0], action[1], -action[2], action[3]]


async def execute_best_action(action):
    """Connect to the robot, execute the action, then reverse it to reset position."""
    if not action or len(action) != 4:
        print("Invalid action vector.")
        return

    print(f"Executing action: {action}")

    # Connect to robot
    opts = RobotClient.Options.with_api_key(api_key=EXEC_API_KEY, api_key_id=EXEC_API_KEY_ID)
    robot = await RobotClient.at_address(EXEC_ROBOT_ADDRESS, opts)
    motor1 = Motor.from_robot(robot=robot, name="motor-1")
    motor2 = Motor.from_robot(robot=robot, name="motor-2")

    # Execute forward action
    m2_revs, m2_rpm, m1_revs, m1_rpm = _scale_action(action)
    print("Executing...")
    await motor2.go_for(rpm=m2_rpm, revolutions=m2_revs)
    await motor1.go_for(rpm=m1_rpm, revolutions=m1_revs)

    # Reset to original position
    m2_revs_r, m2_rpm_r, m1_revs_r, m1_rpm_r = _scale_action(_reverse_action(action))
    print("Resetting position...")
    await motor2.go_for(rpm=m2_rpm_r, revolutions=-m2_revs_r)
    await motor1.go_for(rpm=m1_rpm_r, revolutions=m1_revs_r)

    print("Done.")


# --- Standalone test ---
if __name__ == '__main__':
    import asyncio
    import sys

    if len(sys.argv) != 5:
        print("Usage: python -m robot.execution <slide> <slide_rpm> <rotate> <rotate_rpm>")
        print("  Each value is normalized 0..1  (e.g. 0.5 0.5 0.5 0.5)")
        sys.exit(1)

    action = [float(v) for v in sys.argv[1:5]]
    asyncio.run(execute_best_action(action))
