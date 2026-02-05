# vision.py
import asyncio
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.services.vision import VisionClient

# Pygame game scaling constants - must match your game!
SCALE = 25
WIDTH = 15 * SCALE
HALF_FIELD_LENGTH = 13.5 * SCALE
HEIGHT = HALF_FIELD_LENGTH * 2 + SCALE * 2
GOAL_WIDTH = 3.75 * SCALE

def get_center(bbox):
    return ((bbox.x_min + bbox.x_max) / 2, (bbox.y_min + bbox.y_max) / 2)

def group_by_y(detections, threshold=30):
    """Group bounding boxes by Y-center within a threshold."""
    y_centers = [get_center(d)[1] for d in detections]
    y_centers.sort()

    groups = []
    for y in y_centers:
        matched = False
        for group in groups:
            if abs(y - group['avg']) <= threshold:
                group['vals'].append(y)
                group['avg'] = sum(group['vals']) / len(group['vals'])
                matched = True
                break
        if not matched:
            groups.append({'vals': [y], 'avg': y})
    
    return [round(group['avg'], 1) for group in groups]

def scale_puck_coords(puck_x, puck_y):
    """
    Scale puck coordinates from vision detection to game input space using linear interpolation.

    Updated X-axis:
        real_x = 206.5 → game_x = 150
        real_x = 143.5 → game_x = 22
    """

    # Updated X mapping bounds
    real_x_min = 143.5
    real_x_max = 206.5
    game_x_min = 22
    game_x_max = 150

    # Keep existing Y mapping
    real_y_min = 196.0
    real_y_max = 295.0
    game_y_min = 525
    game_y_max = 600

    # Clamp to vision bounds
    puck_x = max(min(puck_x, real_x_max), real_x_min)
    puck_y = max(min(puck_y, real_y_max), real_y_min)

    # Linear interpolation
    scaled_x = game_x_min + (puck_x - real_x_min) * (game_x_max - game_x_min) / (real_x_max - real_x_min)
    scaled_y = game_y_min + (puck_y - real_y_min) * (game_y_max - game_y_min) / (real_y_max - real_y_min)

    return scaled_x, scaled_y

async def connect():
    opts = RobotClient.Options.with_api_key( 
        api_key='2axyuerwf9mns7s2wg57ez1bi1d7135n',
        api_key_id='f982cb86-7fe9-4ec3-857c-2e9d43c921b8'
    )
    return await RobotClient.at_address('bubble-hockey-macbook.8dfgn52n2e.viam.cloud', opts)

async def get_scaled_puck_coords():
    """
    Connect to the robot, get puck detections, and return the scaled coordinates.
    Returns a tuple of (scaled_x, scaled_y) or (None, None) if no puck is detected.
    """
    machine = await connect()
    
    try:
        camera = Camera.from_robot(machine, "C270")
        vision1 = VisionClient.from_robot(machine, "vision-1")  # Pink
        
        detections1 = await vision1.get_detections_from_camera("C270")
        
        # --- Puck (Pink, class_name = 'rose') ---
        pink_detections = [d for d in detections1 if d.class_name == "rose"]
        
        if pink_detections:
            pink_detections.sort(key=lambda d: d.y_min)
            puck = pink_detections[len(pink_detections) // 2]
            puck_center = get_center(puck)
            real_x, real_y = puck_center[0], puck_center[1]
            print(f"RealPuck x={real_x}, {real_y}")
            
            # ✅ Return scaled coordinates
            scaled_x, scaled_y = scale_puck_coords(puck_center[0], puck_center[1])
            return real_x, scaled_y
        else:
            return None, None
    
    finally:
        await machine.close()

async def main():
    scaled_x, scaled_y = await get_scaled_puck_coords()
    
    if scaled_x is not None and scaled_y is not None:
        print(f"Scaled puck coordinates in game space: x={scaled_x:.1f}, y={scaled_y:.1f}")
    else:
        print("No pink puck detected.")

if __name__ == '__main__':
    asyncio.run(main())