"""Coordinated five-rod relay routine.

Relays the puck through every rod in order:
  Left D -> Right D -> Center -> Left Wing -> Right Wing -> SHOT

Each receiving rod follows the puck's detected position; vision confirms the
puck has reached the next rod before that rod acts. See
docs/superpowers/specs/2026-05-21-coordinated-relay-routine-design.md.
"""

from engine.constants import (
    PlayerID,
    center_x, min_y_center, max_y_center,
    right_wing_x, min_y_right_wing, max_y_right_wing,
    right_d_x, min_y_right_d, max_y_right_d,
    left_d_x, min_y_left_d, max_y_left_d,
    left_wing_x, min_y_left_wing, max_y_left_wing,
)

# Per-rod translation band: t=0 -> min_y, t=1 -> max_y (game pixels).
_ROD_Y_BAND = {
    PlayerID.LEFT_D:     (min_y_left_d, max_y_left_d),
    PlayerID.RIGHT_D:    (min_y_right_d, max_y_right_d),
    PlayerID.CENTER:     (min_y_center, max_y_center),
    PlayerID.LEFT_WING:  (min_y_left_wing, max_y_left_wing),
    PlayerID.RIGHT_WING: (min_y_right_wing, max_y_right_wing),
}


def puck_y_to_t(player_id: PlayerID, puck_y: float) -> float:
    """Map a puck game-y coordinate to a normalized [0, 1] translation for a rod."""
    min_y, max_y = _ROD_Y_BAND[player_id]
    t = (puck_y - min_y) / (max_y - min_y)
    return max(0.0, min(1.0, t))
