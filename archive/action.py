# action.py
from viam.robot.client import RobotClient
from viam.components.motor import Motor
from viam.rpc.dial import Credentials

def scale_vector(scaled):
    motor2_revs = 0.1 + scaled[0] * (-2.3 - 0.1)
    motor2_rpm = scaled[1] * 120
    motor1_revs = scaled[2] * -1 * 0.6667
    motor1_rpm = scaled[3] * 300
    return motor2_revs, motor2_rpm, motor1_revs, motor1_rpm

def reverse_scaled_vector(scaled):
    return [
        scaled[0],
        scaled[1],
        -scaled[2],  # reverse motor1 direction
        scaled[3]
    ]

async def _run_action(best_action):
    if not best_action or len(best_action) != 4:
        print("❌ Invalid best action vector. Exiting.")
        return

    print(f"✅ Executing best action: {best_action}")

    opts = RobotClient.Options.with_api_key(
        api_key='wqcwp98a0ufcgp4xlq6wlaj2q6fq0swu',
        api_key_id='5b00b4f5-d4d4-49a4-a1e1-bb6e90e16e50'
    )
    robot = await RobotClient.at_address('bubble-hockey-pi.8dfgn52n2e.viam.cloud', opts)

    motor1 = Motor.from_robot(robot=robot, name="motor-1")
    motor2 = Motor.from_robot(robot=robot, name="motor-2")

    m2_revs, m2_rpm, m1_revs, m1_rpm = scale_vector(best_action)

    print("--- Executing vector ---")
    await motor2.go_for(rpm=m2_rpm, revolutions=m2_revs)
    await motor1.go_for(rpm=m1_rpm, revolutions=m1_revs)

    reversed_vec = reverse_scaled_vector(best_action)
    m2_revs_r, m2_rpm_r, m1_revs_r, m1_rpm_r = scale_vector(reversed_vec)

    print("--- Reversing vector ---")
    await motor2.go_for(rpm=m2_rpm_r, revolutions=-m2_revs_r)
    await motor1.go_for(rpm=m1_rpm_r, revolutions=m1_revs_r)

    print("✅ All actions completed and reversed.")

async def execute_best_action(best_action):
    # all async code here, call _run_action with await
    await _run_action(best_action)