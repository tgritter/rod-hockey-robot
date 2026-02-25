# ============================================================
#  Robot connection
# ============================================================

# Vision system runs on the MacBook
VISION_ROBOT_ADDRESS = 'bubble-hockey-macbook.8dfgn52n2e.viam.cloud'
VISION_API_KEY       = '2axyuerwf9mns7s2wg57ez1bi1d7135n'
VISION_API_KEY_ID    = 'f982cb86-7fe9-4ec3-857c-2e9d43c921b8'

# Motor control runs on the Pi
EXEC_ROBOT_ADDRESS = 'bubble-hockey-pi.8dfgn52n2e.viam.cloud'
EXEC_API_KEY       = 'wqcwp98a0ufcgp4xlq6wlaj2q6fq0swu'
EXEC_API_KEY_ID    = '5b00b4f5-d4d4-49a4-a1e1-bb6e90e16e50'


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
