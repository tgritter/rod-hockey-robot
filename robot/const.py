import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
#  Robot connection
# ============================================================

# Vision system runs on the MacBook
VISION_ROBOT_ADDRESS = os.getenv('VISION_ROBOT_ADDRESS')
VISION_API_KEY       = os.getenv('VISION_API_KEY')
VISION_API_KEY_ID    = os.getenv('VISION_API_KEY_ID')

# Motor control runs on the Pi
EXEC_ROBOT_ADDRESS = os.getenv('EXEC_ROBOT_ADDRESS')
EXEC_API_KEY       = os.getenv('EXEC_API_KEY')
EXEC_API_KEY_ID    = os.getenv('EXEC_API_KEY_ID')


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
