"""Reward observation after a play executes.

Two reward signals:
  Pass : vision check — did the puck land in the target player's zone?
  Shot : goal sensor  — did it score? (placeholder until sensor is wired)

Reward values:
  pass lands in target zone : +1.0
  pass misses target zone   : -0.3
  goal scored               : +5.0
  no signal / unknown       :  0.0
"""

import asyncio

from engine.constants import PlayerID
from robot.vision import get_puck_camera_coordinates
from robot.playbook import _ZONES

_OBSERVE_DELAY = 0.75  # seconds to wait for the puck to settle after a play

_PASS_HIT   =  1.0
_PASS_MISS  = -0.3
_GOAL_SCORE =  5.0

# Target zone bounds per player in camera pixel space.
# TODO: calibrate these to match the actual camera view.
_CAMERA_ZONES: dict[PlayerID, dict] = {
    PlayerID.CENTER:     {"x_min": 100, "x_max": 400, "y_min": 85,  "y_max": 165},
    PlayerID.RIGHT_WING: {"x_min": 100, "x_max": 400, "y_min": 0,   "y_max": 75},
    PlayerID.LEFT_WING:  {"x_min": 0,   "x_max": 400, "y_min": 110, "y_max": 280},
    PlayerID.RIGHT_D:    {"x_min": 250, "x_max": 485, "y_min": 65,  "y_max": 115},
    PlayerID.LEFT_D:     {"x_min": 250, "x_max": 530, "y_min": 185, "y_max": 235},
}


def _puck_in_zone(cam_x: float, cam_y: float, target: PlayerID) -> bool:
    zone = _CAMERA_ZONES.get(target)
    if zone is None:
        return False
    return zone["x_min"] <= cam_x < zone["x_max"] and zone["y_min"] <= cam_y < zone["y_max"]


async def _read_goal_sensor() -> bool:
    """Return True if a goal was scored. Placeholder until sensor is wired."""
    # TODO: read Viam digital input component for goal sensor
    return False


async def observe_reward(action: dict) -> float:
    """Observe and return the scalar reward for the most recently executed action."""
    await asyncio.sleep(_OBSERVE_DELAY)

    if action["type"] == "shot":
        if await _read_goal_sensor():
            print("Reward: goal scored (+5.0)")
            return _GOAL_SCORE
        return 0.0

    if action["type"] == "pass":
        cam_x, cam_y = await get_puck_camera_coordinates()
        if cam_x is None:
            return 0.0
        if _puck_in_zone(cam_x, cam_y, action["target"]):
            print(f"Reward: pass reached {action['target'].name} (+1.0)")
            return _PASS_HIT
        print(f"Reward: pass missed {action['target'].name} (-0.3)")
        return _PASS_MISS

    return 0.0
