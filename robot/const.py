import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
#  Robot connection
# ============================================================

# Primary part, right wing player + camera. All other parts are configured as
# remotes w/ prefixes.
ROBOT_ADDRESS    = os.getenv('ROBOT_ADDRESS')
ROBOT_API_KEY    = os.getenv('ROBOT_API_KEY')
ROBOT_API_KEY_ID = os.getenv('ROBOT_API_KEY_ID')

# ============================================================
#  Camera → game coordinate mapping
# ============================================================

# Real camera pixel range for the puck's x and y
CAMERA_X_MIN = 143.5
CAMERA_X_MAX = 206.5
CAMERA_Y_MIN = 196.0
CAMERA_Y_MAX = 295.0

# ============================================================
#  Motor config
# ============================================================

# Encoder ticks per full motor rotation (set in Viam motor config)
TICKS_PER_ROTATION = 200

# Lateral slide tick limits (0 = home position)
CENTER_SLIDE_MAX_TICKS = 500   # TODO: calibrate full travel
RW_SLIDE_MAX_TICKS     = 500   # TODO: calibrate full travel


# ============================================================
#  Relay routine
# ============================================================

# Camera component used for puck detection during the relay.
RELAY_CAMERA = "dynamic-crop"

# How close (game pixels) the puck's x must be to a rod to count as "arrived".
RELAY_GATE_TOLERANCE_PX = 30.0   # TODO: calibrate

# Max time to wait for the puck to reach the next rod before aborting (seconds).
RELAY_GATE_TIMEOUT_S = 8.0

# Delay between vision polls while waiting at a gate (seconds).
RELAY_VISION_POLL_INTERVAL_S = 0.3
