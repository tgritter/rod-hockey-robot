"""Run a specific playbook for a single player.

Usage:
  python3 run_play.py --player center --side left
  python3 run_play.py --player right_wing --side left --action shot
  python3 run_play.py --player left_d --side right --action pass

Players: center, left_d, right_d, left_wing, right_wing
Sides:   left, right, bottom_left, bottom_right
Actions: shot, pass  (defaults to the calibrated action for that play)

If the requested action isn't calibrated yet, the robot will run the puck
handling steps then rotate to 180° as a placeholder.
"""

import argparse
import asyncio

from engine.constants import PlayerID
from robot.execution import execute_sequence
from robot.playbook import (
    _CENTER_PLAYBOOK,
    _LEFT_D_PLAYBOOK,
    _RIGHT_D_PLAYBOOK,
    _LEFT_WING_PLAYBOOK,
    _RIGHT_WING_PLAYBOOK,
    find_action,
    get_sequence,
)

_PLAYER_MAP = {
    "center":     PlayerID.CENTER,
    "left_d":     PlayerID.LEFT_D,
    "right_d":    PlayerID.RIGHT_D,
    "left_wing":  PlayerID.LEFT_WING,
    "right_wing": PlayerID.RIGHT_WING,
}

_PLAYBOOKS = {
    "center":     _CENTER_PLAYBOOK,
    "left_d":     _LEFT_D_PLAYBOOK,
    "right_d":    _RIGHT_D_PLAYBOOK,
    "left_wing":  _LEFT_WING_PLAYBOOK,
    "right_wing": _RIGHT_WING_PLAYBOOK,
}

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True, choices=_PLAYER_MAP.keys())
    parser.add_argument("--side", required=True, choices=["left", "right", "bottom_left", "bottom_right"])
    parser.add_argument("--action", default=None, choices=["shot", "pass"],
                        help="Action to execute (default: use the first calibrated action for this play)")
    args = parser.parse_args()

    play = _PLAYBOOKS[args.player][args.side]
    player_id = _PLAYER_MAP[args.player]
    action = find_action(play, args.action)
    sequence = play["puck_handling"] + action["steps"]

    if action["target"] == PlayerID.LEFT_WING:
        await asyncio.gather(
            execute_sequence(sequence, player_id),
            execute_sequence([{"t": 0.6}], PlayerID.LEFT_WING),
        )
        await execute_sequence([{"t": 0}], PlayerID.LEFT_WING)
    else:
        await execute_sequence(sequence, player_id)


if __name__ == "__main__":
    asyncio.run(main())
