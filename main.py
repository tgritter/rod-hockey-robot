"""
Main entry point for the bubble hockey robot.

Pipeline:
  1. Vision   — detect puck position from camera
  2. Playbook — look up calibrated instruction sequence
  3. Execution — send motor commands to the robot

Manual override (skips vision — useful for calibration):
  python main.py --center-left
  python main.py --center-right
  python main.py --rw-shot --left
  python main.py --rw-shot --right
  python main.py --rw-shot --bottom-left
  python main.py --rw-shot --bottom-right
  python main.py --rw-pass --left
  python main.py --rw-pass --right
  python main.py --rw-pass --bottom-left
  python main.py --rw-pass --bottom-right
  python main.py --rd-left
  python main.py --rd-right
  python main.py --ld-left
  python main.py --ld-right

Loop mode (polls vision every 2 seconds):
  python main.py --loop
"""

import asyncio
import argparse
import random

from robot.vision import get_puck_game_coordinates
from robot.playbook import get_instructions, get_rw_sequence, _CENTER_PLAYBOOK, _RIGHT_D_PLAYBOOK, _LEFT_D_PLAYBOOK
from robot.execution import execute_sequence
from engine.constants import PlayerID, min_y_center, TARGET_Y_MAX, min_y_right_wing, max_y_right_wing, center_x, right_wing_x


def parse_args():
    parser = argparse.ArgumentParser(description="Bubble hockey robot")

    parser.add_argument("--loop", action="store_true", help="Poll vision every 2 seconds and act when puck is detected")

    side_group = parser.add_mutually_exclusive_group()
    side_group.add_argument("--left",         action="store_true")
    side_group.add_argument("--right",        action="store_true")
    side_group.add_argument("--bottom-left",  action="store_true")
    side_group.add_argument("--bottom-right", action="store_true")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--center-left",  action="store_true")
    group.add_argument("--center-right", action="store_true")
    group.add_argument("--rw-shot", action="store_true")
    group.add_argument("--rw-pass", action="store_true")
    group.add_argument("--rd-left",  action="store_true")
    group.add_argument("--rd-right", action="store_true")
    group.add_argument("--ld-left",  action="store_true")
    group.add_argument("--ld-right", action="store_true")

    return parser.parse_args()


def _rw_side(args) -> str:
    if args.right:        return "right"
    if args.bottom_left:  return "bottom_left"
    if args.bottom_right: return "bottom_right"
    return "left"


def _rw_action(args, base: str) -> str:
    """Return the action key, using bottom variant when a bottom side is selected."""
    if args.bottom_left or args.bottom_right:
        return f"bottom_{base}"
    return base


async def run_once(args):
    """Run one vision → plan → execute cycle. Returns True if an action was taken."""
    # Manual override — skip vision, infer player from flag
    sequence = None
    player = None
    if args.center_left:  player = PlayerID.CENTER;     sequence = _CENTER_PLAYBOOK["left"]
    if args.center_right: player = PlayerID.CENTER;     sequence = _CENTER_PLAYBOOK["right"]
    if args.rw_shot: player = PlayerID.RIGHT_WING; sequence = get_rw_sequence(_rw_side(args), _rw_action(args, "shot"))
    if args.rw_pass: player = PlayerID.RIGHT_WING; sequence = get_rw_sequence(_rw_side(args), _rw_action(args, "pass"))
    if args.rd_left:  player = PlayerID.RIGHT_D; sequence = _RIGHT_D_PLAYBOOK["left"]
    if args.rd_right: player = PlayerID.RIGHT_D; sequence = _RIGHT_D_PLAYBOOK["right"]
    if args.ld_left:  player = PlayerID.LEFT_D;  sequence = _LEFT_D_PLAYBOOK["left"]
    if args.ld_right: player = PlayerID.LEFT_D;  sequence = _LEFT_D_PLAYBOOK["right"]

    if sequence:
        print(f"Manual override: player={player.name}")
    else:
        # 1. Detect the puck
        puck_x, puck_y = await get_puck_game_coordinates()
        if puck_x is None:
            print("No puck detected.")
            return False
        print(f"Puck detected at: x={puck_x:.1f}, y={puck_y:.1f}")

        # 2. Pick player based on puck x position (rod location), confirm y in range
        _RW_CENTER_MID = (right_wing_x + center_x) / 2   # 125 px
        if puck_x < _RW_CENTER_MID and min_y_right_wing <= puck_y <= max_y_right_wing:
            player = PlayerID.RIGHT_WING
            side = "left" if puck_x < right_wing_x else "right"
            action = random.choice(["shot", "pass"])
            print(f"Right wing: side={side}, action={action}")
            sequence = get_rw_sequence(side, action)
        elif min_y_center <= puck_y <= TARGET_Y_MAX:
            player = PlayerID.CENTER
            sequence = get_instructions(puck_x, puck_y, player)
        else:
            print("Puck out of range — no action.")
            return False

        if not sequence:
            print("No instructions for this position.")
            return False

    # 3. Execute
    await execute_sequence(sequence, player)
    return True


async def main():
    args = parse_args()

    if args.loop:
        print("Loop mode — polling every 2 seconds. Press Ctrl+C to stop.")
        while True:
            await run_once(args)
            await asyncio.sleep(2)
    else:
        await run_once(args)


if __name__ == "__main__":
    asyncio.run(main())
