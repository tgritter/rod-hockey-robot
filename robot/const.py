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
#  Visual-servo relay
# ============================================================

# Camera component used for all relay vision (puck + players).
RELAY_CAMERA = "dynamic-crop"

# Vision retry — the dynamic-crop camera intermittently fails; retry through it.
RELAY_VISION_RETRIES       = 8
RELAY_VISION_RETRY_DELAY_S = 0.4
# Per-call timeout — a detection call can hang if the gRPC channel drops.
RELAY_VISION_CALL_TIMEOUT_S = 8.0

# Contact-based catch: sweep the rod's t in these increments; a puck that moves
# more than RELAY_CONTACT_MOVE_PX (camera pixels) means the player touched it.
RELAY_SWEEP_STEP_T    = 0.1
RELAY_CONTACT_MOVE_PX = 15.0   # TODO: calibrate

# Catch sweep speed — kept low so the player corrals the puck instead of
# batting it away (a fast sweep launches the puck on the low-friction table).
RELAY_CATCH_SPEED_MM_S = 30.0  # TODO: calibrate
