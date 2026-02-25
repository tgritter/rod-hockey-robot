"""
Vision module — detects the puck from the robot's camera and returns
its position in game pixel coordinates.

Connects to the Viam robot, reads detections from the vision service,
and maps the camera-space bounding box center to the game's coordinate system.
"""

import asyncio
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.services.vision import VisionClient

from .const import (
    VISION_ROBOT_ADDRESS, VISION_API_KEY, VISION_API_KEY_ID,
    CAMERA_X_MIN, CAMERA_X_MAX, CAMERA_Y_MIN, CAMERA_Y_MAX,
    GAME_X_MIN, GAME_X_MAX, GAME_Y_MIN, GAME_Y_MAX,
)


def get_center(bbox):
    """Return the (x, y) center of a bounding box."""
    return ((bbox.x_min + bbox.x_max) / 2, (bbox.y_min + bbox.y_max) / 2)


def group_by_y(detections, threshold=30):
    """Group bounding boxes by y-center proximity.

    Returns a sorted list of averaged y-centers, one per cluster.
    Useful for collapsing multiple detections of the same object row.
    """
    y_centers = sorted(get_center(d)[1] for d in detections)
    groups = []
    for y in y_centers:
        for group in groups:
            if abs(y - group['avg']) <= threshold:
                group['vals'].append(y)
                group['avg'] = sum(group['vals']) / len(group['vals'])
                break
        else:
            groups.append({'vals': [y], 'avg': y})
    return [round(g['avg'], 1) for g in groups]


def scale_puck_coords(camera_x, camera_y):
    """Map a puck position from camera space to game pixel space via linear interpolation."""
    camera_x = max(min(camera_x, CAMERA_X_MAX), CAMERA_X_MIN)
    camera_y = max(min(camera_y, CAMERA_Y_MAX), CAMERA_Y_MIN)

    game_x = GAME_X_MIN + (camera_x - CAMERA_X_MIN) * (GAME_X_MAX - GAME_X_MIN) / (CAMERA_X_MAX - CAMERA_X_MIN)
    game_y = GAME_Y_MIN + (camera_y - CAMERA_Y_MIN) * (GAME_Y_MAX - GAME_Y_MIN) / (CAMERA_Y_MAX - CAMERA_Y_MIN)

    return game_x, game_y


async def _connect():
    opts = RobotClient.Options.with_api_key(api_key=VISION_API_KEY, api_key_id=VISION_API_KEY_ID)
    return await RobotClient.at_address(VISION_ROBOT_ADDRESS, opts)


async def get_puck_game_coordinates():
    """Connect to the robot, detect the puck, and return its game-space (x, y).

    Looks for the pink ('rose') detection from vision-1.
    Returns (game_x, game_y) in pixels, or (None, None) if no puck is detected.
    """
    machine = await _connect()
    try:
        vision1 = VisionClient.from_robot(machine, "vision-1")
        detections = await vision1.get_detections_from_camera("C270")

        pink = [d for d in detections if d.class_name == "rose"]
        if not pink:
            return None, None

        # Pick the median detection to reduce noise
        pink.sort(key=lambda d: d.y_min)
        puck = pink[len(pink) // 2]
        camera_x, camera_y = get_center(puck)
        print(f"Camera puck: x={camera_x:.1f}, y={camera_y:.1f}")

        game_x, game_y = scale_puck_coords(camera_x, camera_y)
        return game_x, game_y

    finally:
        await machine.close()


# --- Standalone test ---
async def _main():
    game_x, game_y = await get_puck_game_coordinates()
    if game_x is not None:
        print(f"Game coordinates: x={game_x:.1f}, y={game_y:.1f}")
    else:
        print("No puck detected.")

if __name__ == '__main__':
    asyncio.run(_main())
