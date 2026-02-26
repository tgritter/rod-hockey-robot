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
)
from engine.constants import WIDTH, HEIGHT

# Class name used by the vision service to label field corner markers
_CORNER_CLASS = "lime-green"
_PUCK_CLASS   = "green"


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


def scale_puck_coords(camera_x, camera_y, cam_x_min=CAMERA_X_MIN, cam_x_max=CAMERA_X_MAX,
                      cam_y_min=CAMERA_Y_MIN, cam_y_max=CAMERA_Y_MAX):
    """Map a puck position from camera space to game pixel space.

    Camera is landscape and rotated 90°: camera_x → game_y (long axis),
    camera_y → game_x (short axis). Clamps to camera bounds before mapping.
    """
    camera_x = max(min(camera_x, cam_x_max), cam_x_min)
    camera_y = max(min(camera_y, cam_y_max), cam_y_min)

    game_x = (cam_y_max - camera_y) / (cam_y_max - cam_y_min) * WIDTH
    game_y = (camera_x - cam_x_min) / (cam_x_max - cam_x_min) * HEIGHT

    return game_x, game_y


async def _connect():
    opts = RobotClient.Options.with_api_key(api_key=VISION_API_KEY, api_key_id=VISION_API_KEY_ID)
    return await RobotClient.at_address(VISION_ROBOT_ADDRESS, opts)


async def get_puck_game_coordinates():
    """Connect to the robot, detect the puck, and return its game-space (x, y).

    Fetches puck detections from vision-1 and corner detections from vision-2
    to derive dynamic camera bounds. Falls back to hardcoded bounds if corners
    are not found. Returns (game_x, game_y) in pixels, or (None, None) if no
    puck is detected.
    """
    machine = await _connect()
    try:
        vision1 = VisionClient.from_robot(machine, "vision-1")
        vision2 = VisionClient.from_robot(machine, "vision-2")

        puck_detections, corner_detections = await asyncio.gather(
            vision1.get_detections_from_camera("C270"),
            vision2.get_detections_from_camera("C270"),
        )

        pink = [d for d in puck_detections if d.class_name == _PUCK_CLASS]
        if not pink:
            return None, None

        # Pick the median detection to reduce noise
        pink.sort(key=lambda d: d.y_min)
        puck = pink[len(pink) // 2]
        camera_x, camera_y = get_center(puck)
        print(f"Camera puck: x={camera_x:.1f}, y={camera_y:.1f}")

        bounds = _field_bounds_from_corners(corner_detections)
        if bounds:
            cam_x_min, cam_x_max, cam_y_min, cam_y_max = bounds
        else:
            cam_x_min, cam_x_max = CAMERA_X_MIN, CAMERA_X_MAX
            cam_y_min, cam_y_max = CAMERA_Y_MIN, CAMERA_Y_MAX

        game_x, game_y = scale_puck_coords(camera_x, camera_y, cam_x_min, cam_x_max, cam_y_min, cam_y_max)
        return game_x, game_y

    finally:
        await machine.close()


def _field_bounds_from_corners(detections):
    """Extract camera-space field bounds from corner marker detections.

    Returns (x_min, x_max, y_min, y_max) or None if fewer than 2 corners found.
    """
    corners = [d for d in detections if d.class_name == _CORNER_CLASS]
    if len(corners) < 2:
        return None
    xs = [get_center(d)[0] for d in corners]
    ys = [get_center(d)[1] for d in corners]
    return min(xs), max(xs), min(ys), max(ys)


# --- Standalone test ---
async def _main():
    machine = await _connect()
    try:
        vision1 = VisionClient.from_robot(machine, "vision-1")
        vision2 = VisionClient.from_robot(machine, "vision-2")

        puck_detections, corner_detections = await asyncio.gather(
            vision1.get_detections_from_camera("C270"),
            vision2.get_detections_from_camera("C270"),
        )

        # Report all detections with confidence scores for debugging
        all_detections = puck_detections + corner_detections
        print(f"Raw detections ({len(all_detections)}):")
        for d in all_detections:
            cx, cy = get_center(d)
            print(f"  {d.class_name:12s}  conf={d.confidence:.2f}  center=({cx:.0f}, {cy:.0f})")

        # Derive field bounds from corner markers
        bounds = _field_bounds_from_corners(corner_detections)
        if bounds:
            cam_x_min, cam_x_max, cam_y_min, cam_y_max = bounds
            print(f"Corner-derived camera bounds: x=[{cam_x_min:.1f}, {cam_x_max:.1f}], y=[{cam_y_min:.1f}, {cam_y_max:.1f}]")
        else:
            cam_x_min, cam_x_max = CAMERA_X_MIN, CAMERA_X_MAX
            cam_y_min, cam_y_max = CAMERA_Y_MIN, CAMERA_Y_MAX
            print("No corner markers detected — using hardcoded camera bounds.")

        # Find puck
        pink = [d for d in puck_detections if d.class_name == _PUCK_CLASS]
        if not pink:
            print("No puck detected.")
            return

        pink.sort(key=lambda d: d.y_min)
        puck = pink[len(pink) // 2]
        camera_x, camera_y = get_center(puck)
        print(f"Camera puck:      x={camera_x:.1f}, y={camera_y:.1f}")

        # Map to full game coordinates using field bounds
        # Camera is landscape: camera x → game y (long axis), camera y → game x (short axis)
        game_x = (cam_y_max - camera_y) / (cam_y_max - cam_y_min) * WIDTH
        game_y = (camera_x - cam_x_min) / (cam_x_max - cam_x_min) * HEIGHT
        print(f"Game coordinates: x={game_x:.1f}, y={game_y:.1f}")

    finally:
        await machine.close()

if __name__ == '__main__':
    asyncio.run(_main())
