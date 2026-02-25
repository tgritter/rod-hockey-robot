"""
Main entry point for the bubble hockey robot.

Pipeline:
  1. Vision   — detect puck position from camera (game pixel coordinates)
  2. Planning — select best player and find optimal action
  3. Execution — send motor commands to the robot
"""

import asyncio

from robot.vision import get_puck_game_coordinates
from engine.planner import plan_action
from robot.execution import execute_best_action


async def main():
    # 1. Detect the puck
    puck_x, puck_y = await get_puck_game_coordinates()
    if puck_x is None:
        print("No puck detected.")
        return

    print(f"Puck detected at game coordinates: x={puck_x:.1f}, y={puck_y:.1f}")

    # 2. Plan — find the best action for the detected puck position
    action, player = plan_action(puck_x, puck_y)
    if action is None:
        print("No action found — puck may be out of range.")
        return

    print(f"Action found via {player.name}: {[f'{v:.3f}' for v in action]}")

    # 3. Execute — send the action to the robot motors
    await execute_best_action(action)


if __name__ == "__main__":
    asyncio.run(main())
