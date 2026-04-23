"""Calibrated instruction playbooks for each player.

Instruction format: (motor, revs, rpm)
  motor : "move"   — lateral slide  (motor-movement)
          "rotate" — stick spin     (motor-rotation)
  revs  : exact revolutions (negative = reverse direction)
  rpm   : motor speed in RPM

All values are placeholders — calibrate on hardware.
"""

from engine.constants import (
    PlayerID,
    center_x,
    left_d_x,
    LEFT_WING_SEG_B_X_MID,
    right_d_x,
    right_wing_x,
)


# ── Center player playbook ─────────────────────────────────────────────────────
#
# X-axis: right (puck_x < center_x, closer to 0) vs left (puck_x >= center_x)
# Stickhandle approach: open blade → slide to puck → place → shot.

CENTER_LEFT = [
    ("rotate",   40,   30),
    ("move",    450,  120),
    ("rotate",   40,   30),
    ("rotate", -300,  200),
]

CENTER_RIGHT = [
    ("rotate",  -50,   30),
    ("move",    450,  120),
    ("rotate",  -36,   30),
    ("rotate",  300,  200),
]

_CENTER_PLAYBOOK = {
    "left":  CENTER_LEFT,
    "right": CENTER_RIGHT,
}


# ── Right wing playbook ────────────────────────────────────────────────────────
#
# Two-phase: position sequence + action sequence, concatenated at runtime.
# e.g. RIGHT_WING_LEFT + RIGHT_WING_SHOT

# Position sequences — move puck to sweet spot
RIGHT_WING_LEFT = [
    ("rotate",   40,   30),    # TODO: open blade
    ("move",    450,  120),    # TODO: slide to puck on left
    # ("rotate",  -40,   30),    # TODO: position blade
]

RIGHT_WING_RIGHT = [
    ("rotate",   -40,   30),    # TODO: open blade
    ("move",    420,  120),    # TODO: slide to puck on right
    ("rotate",   -35,   30),    # TODO: open blade
    ("rotate",  115,   30),    # TODO: position blade
    ("move",    30,  120),    # TODO: slide to puck on right
]

RIGHT_WING_BOTTOM_LEFT = [
    ("rotate",   -40,   30),    # TODO: open blade
    ("move",    825,  120),    # TODO: slide to puck on left
    ("rotate",  -75,   30),    # TODO: position blade
    ("move",    -400,  120),    # TODO: slide to puck on left
]

RIGHT_WING_BOTTOM_RIGHT = [
    ("rotate",   40,   30),    # TODO: open blade
    ("move",    825,  120),    # TODO: slide to puck on left
    ("rotate",  70,   30),    # TODO: position blade
    ("move",    -400,  120),    # TODO: slide to puck on left
]

# Action sequences — execute the play
RIGHT_WING_SHOT = [
    ("rotate", -148,   30),    # reposition blade
    ("rotate",  400, 1000),    # shot
]

RIGHT_WING_PASS = [
    ("move",    -20,  120),    # back off
    ("rotate", -300,  135),    # pass
]

RIGHT_WING_BOTTOM_SHOT = [
    ("rotate", 40,  30),    # pass
    ("move",    20,  50),    # back off
    # ("rotate", -148,   30),    # reposition blade
    ("rotate",  -110, 100),    # shot
    # ("move",    -30,  50),    # back off
    #    ("move",    -20,  50),    # back off
    ("rotate",  150, 200),    # shot
]

RIGHT_WING_BOTTOM_PASS = [
    ("move",    -20,  120),    # back off
    ("rotate", -300,  135),    # pass
]

_RIGHT_WING_POSITIONS = {
    "left":         RIGHT_WING_LEFT,
    "right":        RIGHT_WING_RIGHT,
    "bottom_left":  RIGHT_WING_BOTTOM_LEFT,
    "bottom_right": RIGHT_WING_BOTTOM_RIGHT,
}

_RIGHT_WING_ACTIONS = {
    "shot":        RIGHT_WING_SHOT,
    "pass":        RIGHT_WING_PASS,
    "bottom_shot": RIGHT_WING_BOTTOM_SHOT,
    "bottom_pass": RIGHT_WING_BOTTOM_PASS,
}


def get_rw_sequence(side: str, action: str) -> list:
    """Combine a position and action sequence for the right wing."""
    return _RIGHT_WING_POSITIONS[side] + _RIGHT_WING_ACTIONS[action]


# ── Right defenseman playbook ──────────────────────────────────────────────────
#
# Uses center motor. Simple rotate + move for now — calibrate on hardware.

RIGHT_D_LEFT = [
    ("rotate",  40,  30),    # TODO: open blade
    ("move",   450, 120),    # TODO: slide to puck on left
    ("rotate",  -80,  30),    # TODO: position blade
    ("rotate",  160, 200),    # TODO: shot
]

RIGHT_D_RIGHT = [
    ("rotate", -40,  30),    # TODO: open blade
    ("move",   450, 120),    # TODO: slide to puck on right
    ("rotate",  80,  30),    # TODO: position blade
    ("rotate", -160, 200),    # TODO: shot
]

_RIGHT_D_PLAYBOOK = {
    "left":  RIGHT_D_LEFT,
    "right": RIGHT_D_RIGHT,
}


# ── Left defenseman playbook ───────────────────────────────────────────────────
#
# Uses center motor. Simple rotate + move for now — calibrate on hardware.

LEFT_D_LEFT = [
    ("rotate",  -45,  30),    # open blade
    ("move",   2000, 200),    # slide to puck on left
    ("rotate",  80,  30),    # position blade
    ("move",    60, 120),    # nudge forward before pass — tune
    ("rotate", -120, 200),    # pass to center
]

LEFT_D_RIGHT = [
    ("rotate",   60,  30),    # open blade
    ("move",  2000, 200),    # slide to puck on right
    ("rotate",  -80,  30),    # position blade
    ("move",    60, 120),    # nudge forward before pass — tune
    ("rotate",  120, 200),    # pass to center
]

_LEFT_D_PLAYBOOK = {
    "left":  LEFT_D_LEFT,
    "right": LEFT_D_RIGHT,
}


# ── Left wing playbook ───────────────────────────────────────────────────────
#
# Own motors (leftwing-movement, leftwing-rotation).
# Zones are 2D (x + y), but playbook uses simple left/right for now.
# All values are placeholders — calibrate on hardware.

LEFT_WING_LEFT = [
    ("rotate",  60,  30),    # TODO: open blade
    ("move",   550, 120),    # TODO: slide to puck on left
    ("rotate",  150,  30),    # TODO: open blade
    ("move",   50, 120),    # TODO: slide to puck on left
    ("rotate",  -40,  30),    # TODO: open blade
    ("move",   -250, 120),    # TODO: slide to puck on left
    ("move",   40, 120),    # TODO: slide to puck on left
    ("rotate", -200,  1000),    # TODO: position blade
    # ("move",   100, 120),    # TODO: slide to puck on left
    # ("rotate", 160, 200),    # TODO: shot
]

LEFT_WING_RIGHT = [
    ("rotate", -40,  30),    # TODO: open blade
    ("move",   450, 120),    # TODO: slide to puck on right
    ("rotate",  80,  30),    # TODO: position blade
    ("rotate",-160, 200),    # TODO: shot
]

_LEFT_WING_PLAYBOOK = {
    "left":  LEFT_WING_LEFT,
    "right": LEFT_WING_RIGHT,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def _center_side(puck_x: float) -> str:
    return "right" if puck_x < center_x else "left"   # x closer to 0 = right


def get_instructions(puck_x: float, puck_y: float, player_id: PlayerID = PlayerID.CENTER):  # noqa: ARG001
    """Return the calibrated instruction sequence for the given player and puck position."""
    if player_id == PlayerID.CENTER:
        side = _center_side(puck_x)
        print(f"Center side: {side}  (puck_x={puck_x:.0f})")
        return _CENTER_PLAYBOOK[side]

    if player_id == PlayerID.RIGHT_WING:
        side = "left" if puck_x < right_wing_x else "right"
        action = "shot"  # default; caller can override via get_rw_sequence directly
        print(f"Right wing side: {side}  (puck_x={puck_x:.0f})")
        return get_rw_sequence(side, action)

    if player_id == PlayerID.RIGHT_D:
        side = "right" if puck_x < right_d_x else "left"
        print(f"Right D side: {side}  (puck_x={puck_x:.0f})")
        return _RIGHT_D_PLAYBOOK[side]

    if player_id == PlayerID.LEFT_D:
        side = "right" if puck_x < left_d_x else "left"
        print(f"Left D side: {side}  (puck_x={puck_x:.0f})")
        return _LEFT_D_PLAYBOOK[side]

    if player_id == PlayerID.LEFT_WING:
        side = "right" if puck_x < LEFT_WING_SEG_B_X_MID else "left"
        print(f"Left Wing side: {side}  (puck_x={puck_x:.0f})")
        return _LEFT_WING_PLAYBOOK[side]

    return None
 
 
 
# RIGHT_WING_PASS = [
#     ("rotate",  0.2,   30),    # TODO: open blade to horizontal
#     ("move",    2.25, 120),    # TODO: slide to puck
#     ("move",   -0.1,  120),    # TODO: back off
#     ("rotate", -1.5,  135),    # TODO: pass
# ]

# RIGHT_WING_SHOT = [
#     ("rotate",  0.2,   30),    # TODO: open blade to horizontal
#     ("move",    2.1,  120),    # TODO: slide to puck
#     ("rotate", -0.74,   30),    # TODO: reposition blade
#     # ("move",    0.25, 120),    # TODO: back off
#     ("rotate",  2.0, 1000),    # shot
# ]
