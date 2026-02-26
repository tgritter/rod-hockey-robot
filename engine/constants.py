"""
Field dimensions, player positions, zone boundaries, and target coordinates
for the bubble hockey robot.

All pixel values assume SCALE = 25 px/inch.
Field is 18 × 33.5 inches → 450 × 837.5 px.

Target index reference (used by scoring and search functions):
  0  — Center red box       (goal-scoring zone)
  1  — Left Wing red box    (Left Wing target zone)
  2  — Right D zone 2 center
  3  — Yellow box           (between center y-start and red box top)
  4  — Left D zone 2 center
  5  — Left D zone 3 center
  6  — Right Wing zone 2 center
"""

from enum import IntEnum


# ============================================================
#  Player identifiers
# ============================================================

class PlayerID(IntEnum):
    CENTER     = 0
    RIGHT_WING = 1
    LEFT_WING  = 2
    RIGHT_D    = 3
    LEFT_D     = 4


# ============================================================
#  Scale & field dimensions
# ============================================================

SCALE = 25  # pixels per inch

# Physical field size
WIDTH  = 18   * SCALE   # 450 px
HEIGHT = 33.5 * SCALE   # 837.5 px

# Physical constants (inches → pixels)
GOAL_WIDTH                = 3.75 * SCALE
HALF_FIELD_LENGTH         = 16.75 * SCALE
PLAYER_DIAMETER           = 1    * SCALE
STICK_LENGTH              = 1.75 * SCALE
PUCK_DIAMETER             = 1    * SCALE
PLAYER_MIN_DIST_FROM_GOAL = 4    * SCALE
BEHIND_GOAL_SPACE         = 5.5  * SCALE

# Goal line y-coordinates
TOP_GOAL_Y    = BEHIND_GOAL_SPACE
BOTTOM_GOAL_Y = HEIGHT - BEHIND_GOAL_SPACE - SCALE / 2


# ============================================================
#  Colors
# ============================================================

WHITE = (255, 255, 255)
BLUE  = (66,  135, 245)
RED   = (245, 66,  66)
BLACK = (0,   0,   0)


# ============================================================
#  Player home ranges  (inches → pixels)
# ============================================================

# Center
min_y_center = 15   * SCALE    # 375 px
max_y_center = 24.5 * SCALE    # 612.5 px
center_x     = 8    * SCALE    # 200 px

# Right Wing
min_y_right_wing = 14.5 * SCALE   # 362.5 px
max_y_right_wing = 31 * SCALE   # 775 px
right_wing_x     = 2  * SCALE   # 50 px

# Right D
min_y_right_d = 6   * SCALE   # 150 px
max_y_right_d = 14  * SCALE   # 350 px
right_d_x     = 5.5 * SCALE   # 137.5 px

# Left D
min_y_left_d = 1.5 * SCALE   # 37.5 px
max_y_left_d = 18  * SCALE   # 450 px
left_d_x     = 13  * SCALE   # 325 px


# ============================================================
#  Scoring zones
# ============================================================

# Center red box  (target_idx 0)
TARGET_X_MIN = 160
TARGET_X_MAX = 270
TARGET_Y_MIN = 530
TARGET_Y_MAX = 600

# Left Wing red box  (target_idx 1)
TARGET_X_MIN_P2 = 10
TARGET_X_MAX_P2 = 5.25 * SCALE   # 131.25 px
TARGET_Y_MIN_P2 = 530
TARGET_Y_MAX_P2 = 775


# ============================================================
#  Right Wing zones  (thirds of vertical range)
#
#  Zone 1: min_y → ZONE1_MAX_Y    aim → yellow box
#  Zone 2: ZONE1_MAX_Y → ZONE2_MAX_Y  aim → center red box
#  Zone 3: ZONE2_MAX_Y → max_y    aim → center of zone 2
# ============================================================

RIGHT_WING_ZONE_THIRD  = (max_y_right_wing - min_y_right_wing) / 3
RIGHT_WING_ZONE1_MAX_Y = min_y_right_wing +     RIGHT_WING_ZONE_THIRD   # ~508 px
RIGHT_WING_ZONE2_MAX_Y = min_y_right_wing + 2 * RIGHT_WING_ZONE_THIRD   # ~642 px

# Target: center of zone 2  (target_idx 6)
TARGET_RIGHT_WING_Z2_X     = right_wing_x
TARGET_RIGHT_WING_Z2_Y     = (RIGHT_WING_ZONE1_MAX_Y + RIGHT_WING_ZONE2_MAX_Y) / 2   # ~575 px
TARGET_RIGHT_WING_Z2_X_MIN = 0
TARGET_RIGHT_WING_Z2_X_MAX = right_wing_x + 2 * SCALE
TARGET_RIGHT_WING_Z2_Y_MIN = RIGHT_WING_ZONE1_MAX_Y
TARGET_RIGHT_WING_Z2_Y_MAX = RIGHT_WING_ZONE2_MAX_Y


# ============================================================
#  Right D zones  (split at midpoint)
#
#  Zone 1: min_y → MID_Y   aim → center of zone 2
#  Zone 2: MID_Y → max_y   aim → yellow box
# ============================================================

RIGHT_D_ZONE_MID_Y = (min_y_right_d + max_y_right_d) / 2   # 250 px

# Target: center of zone 2  (target_idx 2)
TARGET_RIGHT_D_Z2_X     = right_d_x
TARGET_RIGHT_D_Z2_Y     = (RIGHT_D_ZONE_MID_Y + max_y_right_d) / 2   # 300 px
TARGET_RIGHT_D_Z2_X_MIN = right_d_x - 1.5 * SCALE
TARGET_RIGHT_D_Z2_X_MAX = right_d_x + 1.5 * SCALE
TARGET_RIGHT_D_Z2_Y_MIN = RIGHT_D_ZONE_MID_Y
TARGET_RIGHT_D_Z2_Y_MAX = max_y_right_d


# ============================================================
#  Left D zones  (thirds of vertical range)
#
#  Zone 1: min_y → ZONE1_MAX_Y    aim → middle of zone 2
#  Zone 2: ZONE1_MAX_Y → ZONE2_MAX_Y  aim → middle of zone 3
#  Zone 3: ZONE2_MAX_Y → max_y    aim → yellow box
# ============================================================

LEFT_D_ZONE_THIRD  = (max_y_left_d - min_y_left_d) / 3
LEFT_D_ZONE1_MAX_Y = min_y_left_d +     LEFT_D_ZONE_THIRD   # 175 px
LEFT_D_ZONE2_MAX_Y = min_y_left_d + 2 * LEFT_D_ZONE_THIRD   # 312.5 px

# Target: center of zone 2  (target_idx 4)
TARGET_LEFT_D_Z2_X     = left_d_x
TARGET_LEFT_D_Z2_Y     = (LEFT_D_ZONE1_MAX_Y + LEFT_D_ZONE2_MAX_Y) / 2   # ~244 px
TARGET_LEFT_D_Z2_X_MIN = left_d_x - 1.5 * SCALE
TARGET_LEFT_D_Z2_X_MAX = left_d_x + 1.5 * SCALE
TARGET_LEFT_D_Z2_Y_MIN = LEFT_D_ZONE1_MAX_Y
TARGET_LEFT_D_Z2_Y_MAX = LEFT_D_ZONE2_MAX_Y

# Target: center of zone 3  (target_idx 5)
TARGET_LEFT_D_Z3_X     = left_d_x
TARGET_LEFT_D_Z3_Y     = (LEFT_D_ZONE2_MAX_Y + max_y_left_d) / 2   # ~381 px
TARGET_LEFT_D_Z3_X_MIN = left_d_x - 1.5 * SCALE
TARGET_LEFT_D_Z3_X_MAX = left_d_x + 1.5 * SCALE
TARGET_LEFT_D_Z3_Y_MIN = LEFT_D_ZONE2_MAX_Y
TARGET_LEFT_D_Z3_Y_MAX = max_y_left_d


# ============================================================
#  Left Wing zones  (2D — depends on both y and x)
#
#  Zone 1: y <= BOTTOM_GOAL_Y                aim → center red box
#  Zone 2: y > BOTTOM_GOAL_Y, x > SEG_B_MID  aim → middle of zone 1
#  Zone 3: y > BOTTOM_GOAL_Y, x <= SEG_B_MID aim → Right Wing zone 3 center
# ============================================================

# x-midpoint of segment B (the behind-net horizontal sweep)
LEFT_WING_SEG_B_X_MID = (16.5 * SCALE + 5.72 * SCALE) / 2   # 277.75 px

# Zone 1 target: center of the vertical-sweep region
LEFT_WING_Z1_TARGET_X = (16.0 + 16.5) / 2 * SCALE           # 406.25 px
LEFT_WING_Z1_TARGET_Y = (21.5 * SCALE + BOTTOM_GOAL_Y) / 2  # ~587.5 px

# Zone 3 target: center of Right Wing zone 3
LEFT_WING_Z3_TARGET_X = right_wing_x
LEFT_WING_Z3_TARGET_Y = (RIGHT_WING_ZONE2_MAX_Y + max_y_right_wing) / 2   # ~708 px


# ============================================================
#  Yellow box target  (target_idx 3)
#  Sits between the center's y-start and the red box top.
#  Shared aim for Right D zone 2 and Left D zone 3.
# ============================================================

YELLOW_BOX_TARGET_X     = (TARGET_X_MIN + TARGET_X_MAX) / 2   # 215 px
YELLOW_BOX_TARGET_Y     = (min_y_center + TARGET_Y_MIN) / 2   # 446 px
YELLOW_BOX_TARGET_X_MIN = TARGET_X_MIN
YELLOW_BOX_TARGET_X_MAX = TARGET_X_MAX
YELLOW_BOX_TARGET_Y_MIN = min_y_center
YELLOW_BOX_TARGET_Y_MAX = TARGET_Y_MIN
