import asyncio
import numpy as np
from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.services.vision import VisionClient

async def connect():
    """Connect to the Viam robot."""
    # Assuming API key and ID are correct from your snippet
    opts = RobotClient.Options.with_api_key( 
        api_key='2axyuerwf9mns7s2wg57ez1bi1d7135n', # Use your actual key
        api_key_id='f982cb86-7fe9-4ec3-857c-2e9d43c921b8' # Use your actual key ID
    )
    return await RobotClient.at_address('bubble-hockey-macbook.8dfgn52n2e.viam.cloud', opts)

def get_center(bbox):
    """Get the center point of a bounding box."""
    return ((bbox.x_min + bbox.x_max) / 2, (bbox.y_min + bbox.y_max) / 2)

def order_corners(corner_points):
    """
    Order 4 corner points as: top-left, top-right, bottom-right, bottom-left
    """
    if len(corner_points) != 4:
        return None
    
    sorted_by_y = sorted(corner_points, key=lambda p: p[1])
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]
    
    top_two.sort(key=lambda p: p[0])
    top_left, top_right = top_two
    
    bottom_two.sort(key=lambda p: p[0])
    bottom_left_of_pair, bottom_right_of_pair = bottom_two
    
    return [top_left, top_right, bottom_right_of_pair, bottom_left_of_pair]

def compute_homography(src_points, dst_points):
    """
    Compute homography matrix from source points to destination points.
    Both should be lists of 4 (x, y) tuples.
    """
    if len(src_points) != 4 or len(dst_points) != 4:
        return None
    
    A = []
    for i in range(4):
        x, y = src_points[i]
        u, v = dst_points[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    U, s, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    return H

def apply_homography(point, H):
    """
    Apply homography transformation to a point.
    point: (x, y) tuple
    H: 3x3 homography matrix
    Returns: (x', y') transformed point
    """
    x, y = point
    src = np.array([x, y, 1])
    dst = H @ src
    if dst[2] != 0:
        return (dst[0] / dst[2], dst[1] / dst[2])
    else:
        return None

def calculate_center_from_corners(corner_markers):
    """
    Calculate the center point from the 4 corner markers
    """
    if len(corner_markers) != 4:
        return None
    
    corner_centers = [get_center(marker) for marker in corner_markers]
    
    # Average all corner positions to get approximate center
    center_x = sum(pos[0] for pos in corner_centers) / 4
    center_y = sum(pos[1] for pos in corner_centers) / 4
    
    return (center_x, center_y)

def normalize_ball_position_with_center_calibration(pink_detection, corner_markers, debug_output=True):
    """
    Enhanced normalization using corner markers with center-based calibration
    """
    if not pink_detection or len(corner_markers) != 4:
        print(f"Missing required detections. Need 4 corners, got {len(corner_markers)}")
        return None

    corner_centers = [get_center(marker) for marker in corner_markers]
    ordered_corners = order_corners(corner_centers)
    if ordered_corners is None:
        print("Failed to order corner markers")
        return None

    target_corners = [
        (0, 0),         # Top-left  
        (33.25, 0),     # Top-right
        (33.25, 18),    # Bottom-right
        (0, 18)         # Bottom-left
    ]

    if debug_output:
        print(f"\n--- DEBUG: Corner Mapping ---")
        labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        for i, (corner, target, label) in enumerate(zip(ordered_corners, target_corners, labels)):
            print(f"{label}: Camera({corner[0]:.1f}, {corner[1]:.1f}) -> Target{target}")

    H = compute_homography(ordered_corners, target_corners)
    if H is None:
        print("Failed to compute homography matrix")
        return None

    # Calculate center from corner markers for calibration
    calculated_center = calculate_center_from_corners(corner_markers)
    calibration_offset_x = 0
    calibration_offset_y = 0
    
    if calculated_center:
        center_in_inches = apply_homography(calculated_center, H)
        
        if center_in_inches:
            # The center should be at (16.625, 9.0) inches
            expected_center = (33.25/2, 18/2)
            calibration_offset_x = expected_center[0] - center_in_inches[0]
            calibration_offset_y = expected_center[1] - center_in_inches[1]
            
            if debug_output:
                print(f"\n--- CENTER-BASED CALIBRATION ---")
                print(f"Calculated center from corners: ({calculated_center[0]:.1f}, {calculated_center[1]:.1f}) pixels")
                print(f"Center transformed to: ({center_in_inches[0]:.3f}, {center_in_inches[1]:.3f}) inches")
                print(f"Expected center at: ({expected_center[0]:.3f}, {expected_center[1]:.3f}) inches")
                print(f"Calibration offset: ({calibration_offset_x:.3f}, {calibration_offset_y:.3f}) inches")

    # Transform puck position
    ball_center = get_center(pink_detection)
    puck_in_inches = apply_homography(ball_center, H)
    if puck_in_inches is None:
        return None

    # Apply calibration offset to puck position
    puck_x = puck_in_inches[0] + calibration_offset_x
    puck_y = puck_in_inches[1] + calibration_offset_y
    
    x_norm = puck_x / 33.25
    y_norm = puck_y / 18.0

    if debug_output:
        print(f"\n--- PUCK POSITION CALCULATION ---")
        print(f"Raw puck position: ({puck_in_inches[0]:.3f}, {puck_in_inches[1]:.3f}) inches")
        print(f"Calibrated puck position: ({puck_x:.3f}, {puck_y:.3f}) inches")

    # Swap X and Y coordinates to match your expected orientation
    return (y_norm, x_norm)

async def get_puck_game_coordinates(debug_output=True):
    machine = await connect()
    try:
        camera = Camera.from_robot(machine, "C270")
        vision2 = VisionClient.from_robot(machine, "vision-2") 
        vision3 = VisionClient.from_robot(machine, "vision-3") 
        
        pink_detections = await vision2.get_detections_from_camera("C270")
        yellow_detections = await vision3.get_detections_from_camera("C270")
        
        pink_pucks = [d for d in pink_detections if d.class_name == "green"]
        pink_puck = pink_pucks[0] if pink_pucks else None
        
        corner_markers = [d for d in yellow_detections if d.class_name == "lime-green"]
        
        if debug_output:
            print(f"\n--- Corner Markers Detected: {len(corner_markers)} ---")
            temp_corner_centers_for_log = []
            for marker in corner_markers:
                 temp_corner_centers_for_log.append(get_center(marker))
            for i, center_coord in enumerate(temp_corner_centers_for_log):
                print(f"Raw Corner {i+1} center: ({center_coord[0]:.1f}, {center_coord[1]:.1f})")

        if pink_puck and len(corner_markers) == 4:
            pink_center = get_center(pink_puck)
            if debug_output:
                print(f"\n--- Pink Puck (camera pixels) ---")
                print(f"Center: ({pink_center[0]:.1f}, {pink_center[1]:.1f})")
            
            normalized_coords = normalize_ball_position_with_center_calibration(
                pink_puck, corner_markers, debug_output
            )
            
            if normalized_coords:
                x_norm, y_norm = normalized_coords
                if debug_output:
                    print(f"\n--- FINAL NORMALIZED COORDINATES (0-1) ---")
                    print(f"Puck position: ({x_norm:.3f}, {y_norm:.3f})")
                    print(f"Percentage: ({x_norm*100:.1f}%, {y_norm*100:.1f}%)")
                return (x_norm, y_norm)
            else:
                if debug_output:
                    print("Could not calculate normalized coordinates.")
                return None
        else:
            if debug_output:
                if not pink_puck:
                    print("\nNo pink puck detected (or incorrect class_name filter)")
                if len(corner_markers) != 4:
                    print(f"\nInsufficient corner markers detected: {len(corner_markers)} (need 4)")
            return None
    finally:
        await machine.close()

if __name__ == '__main__':
    coords = asyncio.run(get_puck_game_coordinates())
    if coords:
        print(f"Final Normalized puck coordinates: ({coords[0]:.3f}, {coords[1]:.3f})")
    else:
        print("Failed to get normalized puck coordinates.")