# ============================================================
#  Robot connection
# ============================================================

# Vision system runs on the MacBook
VISION_ROBOT_ADDRESS = 'vision-main.wpth7nhx2w.viam.cloud'
VISION_API_KEY       = 'gc81nfdx1kgokgipl4hu0lcvylob6to0'
VISION_API_KEY_ID    = '40bca621-0c69-4367-b566-c7d58625300e'

# Motor control runs on the Pi
EXEC_ROBOT_ADDRESS = 'rig2-main.wpth7nhx2w.viam.cloud'
EXEC_API_KEY       = '9q2hqp884rvbhuo53g229pni5pqzl9dg'
EXEC_API_KEY_ID    = '2593b412-0ea9-4881-830b-f790c6f6a2c8'


# ============================================================
#  Camera → game coordinate mapping
# ============================================================

# Real camera pixel range for the puck's x and y
CAMERA_X_MIN = 143.5
CAMERA_X_MAX = 206.5
CAMERA_Y_MIN = 196.0
CAMERA_Y_MAX = 295.0

# Corresponding game pixel range
GAME_X_MIN = 22
GAME_X_MAX = 150
GAME_Y_MIN = 525
GAME_Y_MAX = 600


# ============================================================
#  Motor scaling
# ============================================================

# Motor 2: linear (slide) movement. Revolutions are interpolated across [MIN, MAX].
MOTOR2_REVS_MIN = 0.1
MOTOR2_REVS_MAX = -2.3
MOTOR2_RPM_MAX  = 120

# Motor 1: rotation. Scaled by a fixed gear ratio and max RPM.
MOTOR1_REVS_SCALE = 2 / 3   # ~0.6667
MOTOR1_RPM_MAX    = 300
