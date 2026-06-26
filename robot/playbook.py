"""Calibrated instruction playbooks for each player.

Each step is a dict matching the hockey-player module's DoCommand payload:
  {"t": 0.5, "r": 90, "rpm": 30, "speed_mm_per_sec": 100}

Fields (all optional; omit to skip an axis or use config defaults):
  t                 : translation target, normalized over [min_translation_mm,
                      max_translation_mm]. Range [0, 1].
  r                 : rotation target in degrees. Range [0, 360].
  rpm               : rotation speed (defaults from component config).
  speed_mm_per_sec  : translation speed (defaults from component config).

Each play is a dict:
  puck_handling : steps to position the puck before the action
  actions       : list of possible actions, each with:
                    type   — "shot" or "pass"
                    target — PlayerID of the pass recipient, or None for shots
                    steps  — motor commands to execute the action

All values below are placeholders -- calibrate on hardware.
"""

from engine.constants import PlayerID
from . import zones

_FALLBACK_ACTION = {"name": None, "type": None, "target": None, "steps": [{"r": 180}]}


def find_action(play: dict, action_type: str = None) -> dict:
    """Return the matching action from the play, or a placeholder if not found.

    If action_type is None, returns the first action in the list.
    """
    actions = play["actions"]
    if action_type is None:
        return actions[0]
    match = next((a for a in actions if a["type"] == action_type), None)
    if match is None:
        print(f"No calibrated '{action_type}' for this play — using placeholder action.")
        return _FALLBACK_ACTION
    return match


def get_sequence(play: dict, action_type: str = None) -> list:
    """Return the full flat step sequence (puck_handling + action steps)."""
    action = find_action(play, action_type)
    return play["puck_handling"] + action["steps"]


# ── Center player playbook ─────────────────────────────────────────────────────
#
# Side (left/right, and the middle_* bands) comes from the matched zone polygon.

CENTER_LEFT = {
    "puck_handling": [
        {"t": 0.95, "r": 100},  # TODO: calibrate -- center, puck on left
    ],
    "actions": [
        {"name": "v0", "type": "shot", "target": None, "steps": [
            {"r": 300, "rpm": 400, "direction": "ccw"},
        ]},
    ],
}

CENTER_MIDDLE_LEFT = {
    "puck_handling": [
        {"t": 0.95, "r": 100},
    ],
    "actions": [
        {"name": "v0", "type": "shot", "target": None, "steps": [
            {"r": 220, "rpm": 20, "direction": "ccw"},
            {"r": 0, "rpm": 250, "direction": "cw"},
            {"r": 90, "rpm": 1000, "direction": "cw"},
        ]},
    ],
}

CENTER_MIDDLE_RIGHT = {
    "puck_handling": [
        {"t": 0.95, "r": 260},
    ],
    "actions": [
        {"name": "v0", "type": "shot", "target": None, "steps": [
            {"r": 150, "rpm": 20, "direction": "cw"},
            {"r": 0, "rpm": 250, "direction": "ccw"},
            {"r": 270, "rpm": 1000, "direction": "ccw"},
        ]},
    ],
}

CENTER_RIGHT = {
    "puck_handling": [
        {"t": 0.95, "r": 260},  # TODO: calibrate -- center, puck on right
    ],
    "actions": [
        {"name": "v0", "type": "shot", "target": None, "steps": [
            {"r": 60, "rpm": 400, "direction": "cw"},
        ]},
    ],
}

_CENTER_PLAYBOOK = {
    "left":         CENTER_LEFT,
    "middle_left":  CENTER_MIDDLE_LEFT,
    "middle_right": CENTER_MIDDLE_RIGHT,
    "right":        CENTER_RIGHT,
}


# ── Right wing playbook ────────────────────────────────────────────────────────

RIGHT_WING_LEFT = {
    "puck_handling": [
        {"t": 0.5, "r": 125, "speed_mm_per_sec": 10000},  # TODO: calibrate -- right wing position, puck on left
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.CENTER, "steps": [
            {"t": 0.485, "r": 180, "direction": "ccw", "speed_mm_per_sec": 10000},
            {"r": 0, "rpm": 150},
        ]},
    ],
}

RIGHT_WING_RIGHT = {
    "puck_handling": [
        {"t": 0.5, "r": 265, "speed_mm_per_sec": 10000},  # TODO: calibrate -- right wing position, puck on right
    ],
    "actions": [
        {"name": "v0", "type": "shot", "target": None, "steps": [
            {"r": 60, "rpm": 300, "direction": "ccw"},
        ]},
    ],
}

RIGHT_WING_BOTTOM_LEFT = {
    "puck_handling": [
        {"t": 1, "r": 0.0, "speed_mm_per_sec": 10000},  # TODO: calibrate -- right wing position, puck bottom-left
        {"r": 80, "direction": "ccw"},
        {"t": 0.5, "speed_mm_per_sec": 10000},
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.CENTER, "steps": [
            {"t": 0.525, "r": 270, "direction": "cw", "speed_mm_per_sec": 10000},
            {"t": 0.5, "r": 180, "rpm": 400, "direction": "cw", "speed_mm_per_sec": 10000},
        ]},
    ],
}

RIGHT_WING_BOTTOM_RIGHT = {
    "puck_handling": [
        {"t": 1, "r": 0.0, "speed_mm_per_sec": 10000},  # TODO: calibrate -- right wing position, puck bottom-right
        {"r": 295, "direction": "cw"},
        {"t": 0.50, "speed_mm_per_sec": 10000},
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.CENTER, "steps": [
            {"r": 60, "rpm": 260},
        ]},
    ],
}

_RIGHT_WING_PLAYBOOK = {
    "left":         RIGHT_WING_LEFT,
    "right":        RIGHT_WING_RIGHT,
    "bottom_left":  RIGHT_WING_BOTTOM_LEFT,
    "bottom_right": RIGHT_WING_BOTTOM_RIGHT,
}


# ── Right defenseman playbook ──────────────────────────────────────────────────

RIGHT_D_LEFT = {
    "puck_handling": [
        {"r": 100},
        {"t": 0.95},
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.RIGHT_WING, "steps": [
            {"r": 300, "direction": "cw", "rpm": 220},
        ]},
    ],
}

RIGHT_D_RIGHT = {
    "puck_handling": [
        {"t": 1, "r": 260},  # TODO: calibrate -- right D, puck on left
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.CENTER, "steps": [
            {"r": 60, "direction": "ccw", "rpm": 200},
        ]},
    ],
}

_RIGHT_D_PLAYBOOK = {
    "left":  RIGHT_D_LEFT,
    "right": RIGHT_D_RIGHT,
}


# ── Left defenseman playbook ───────────────────────────────────────────────────

LEFT_D_LEFT = {
    "puck_handling": [
        {"t": 0.8, "r": 115},  # TODO: calibrate -- left D, puck on left (pass to center)
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.CENTER, "steps": [
            {"r": 300, "direction": "cw", "rpm": 220},
        ]},
    ],
}

LEFT_D_RIGHT = {
    "puck_handling": [
        {"t": 0.8, "r": 255, "direction": "ccw"},  # TODO: calibrate -- left D, puck on right (pass to left wing)
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.LEFT_WING, "steps": [
            {"r": 60, "direction": "ccw", "rpm": 220},
        ]},
    ],
}

_LEFT_D_PLAYBOOK = {
    "left":  LEFT_D_LEFT,
    "right": LEFT_D_RIGHT,
}


# ── Left wing playbook ─────────────────────────────────────────────────────────
#
# Zones are 2D (x + y), but playbook uses simple left/right for now.

LEFT_WING_LEFT = {
    "puck_handling": [
        {"t": 0.2, "r": 120, "speed_mm_per_sec": 10000},  # TODO: calibrate -- left wing, puck on left
    ],
    "actions": [
        {"name": "v0", "type": "shot", "target": None, "steps": [
            {"r": 300, "rpm": 400, "direction": "cw"},
        ]},
    ],
}

LEFT_WING_RIGHT = {
    "puck_handling": [
        {"t": 0.35, "r": 260, "speed_mm_per_sec": 10000},  # TODO: calibrate -- left wing, puck on right
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.CENTER, "steps": [
            {"t": 0.34, "r": 90, "direction": "cw", "speed_mm_per_sec": 10000},
            {"r": 300, "rpm": 1000, "direction": "cw"},
        ]},
    ],
}

LEFT_WING_BOTTOM_LEFT = {
    "puck_handling": [
        {"t": 0.5, "r": 110, "speed_mm_per_sec": 10000},  # TODO: calibrate -- left wing, puck bottom-left
        {"t": 0.75, "r": 150, "speed_mm_per_sec": 10000},
        {"t": 1, "r": 120, "direction": "ccw", "speed_mm_per_sec": 10000},
        {"r": 100, "direction": "ccw"},
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.RIGHT_WING, "steps": [
            {"r": 300, "rpm": 200, "direction": "cw"},
        ]},
    ],
}

LEFT_WING_BOTTOM_RIGHT = {
    "puck_handling": [
        {"t": 0.5, "r": 270, "speed_mm_per_sec": 10000},  # TODO: calibrate -- left wing, puck bottom-right
        {"t": 1, "r": 320, "direction": "cw", "speed_mm_per_sec": 10000},
    ],
    "actions": [
        {"name": "v0", "type": "pass", "target": PlayerID.RIGHT_WING, "steps": [
            {"r": 100, "rpm": 300, "direction": "ccw"},
        ]},
    ],
}

_LEFT_WING_PLAYBOOK = {
    "left":         LEFT_WING_LEFT,
    "right":        LEFT_WING_RIGHT,
    "bottom_left":  LEFT_WING_BOTTOM_LEFT,
    "bottom_right": LEFT_WING_BOTTOM_RIGHT,
}


# ── Public API ─────────────────────────────────────────────────────────────────

_PLAYBOOK_MAP = {
    PlayerID.CENTER:     _CENTER_PLAYBOOK,
    PlayerID.RIGHT_WING: _RIGHT_WING_PLAYBOOK,
    PlayerID.RIGHT_D:    _RIGHT_D_PLAYBOOK,
    PlayerID.LEFT_D:     _LEFT_D_PLAYBOOK,
    PlayerID.LEFT_WING:  _LEFT_WING_PLAYBOOK,
}


def select_playbook(u: float, v: float):
    """Return (PlayerID, side, play) for the puck at normalized (u, v), or (None, None, None).

    Side comes straight from the matched polygon. CENTER's middle_left/middle_right
    plays are available in _CENTER_PLAYBOOK but only reachable once middle polygons
    are added to zones.json.
    """
    player_id, side = zones.select(u, v)
    if player_id is None:
        return None, None, None
    print(f"{player_id.name} side: {side}  (u={u:.3f}, v={v:.3f})")
    return player_id, side, _PLAYBOOK_MAP[player_id][side]
