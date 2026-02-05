import asyncio
import sys
from vision import get_puck_game_coordinates
from calc import find_best_action
from action import execute_best_action

async def main():
    # 1. Detect the puck
    coords = await get_puck_game_coordinates()
    if coords:
        x, y = coords  # Both x and y are between 0 and 1
        print(f"Ball at ({x:.3f}, {y:.3f})")
        scaled_x = 450 - (x * 450)
        scaled_y = y * 787.5
        print(f"X Axis:({scaled_x:.3f})")
        print(f"Y Axis:({scaled_y:.3f})")
        # 2. Calculate best action
        best_action = find_best_action(scaled_x, scaled_y)
        if best_action:
            print(f"Best action found: {best_action}")
            # 2. Execute best action
            await execute_best_action(best_action)
            sys.exit()    
        else:
            print("No successful action found.")
    
    # if not coords:
    #     print("Failed to detect puck. Please check camera and markers.")
    #     return 1
    
    # scaled_x, scaled_y = coords

    if scaled_x is not None and scaled_y is not None:
        print(f"✅ Puck coordinates: x={scaled_x:.1f}, y={scaled_y:.1f}")
        # 2. Calculate best action
        best_action = find_best_action(scaled_x, 600.0)
        if best_action:
            print(f"Best action found: {best_action}")
            # 2. Execute best action
            await execute_best_action(best_action)
            sys.exit()    
        else:
            print("No successful action found.")
    else:
        print("❌ No puck detected")

if __name__ == "__main__":
    asyncio.run(main())