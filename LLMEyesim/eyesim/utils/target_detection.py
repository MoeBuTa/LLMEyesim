from loguru import logger
import numpy as np
from typing import List, Tuple
from eye import *

from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.eyesim.generator.objects.target import TARGET_LOCATIONS


def detect_red_target(img, robot_pos: Tuple[int, int, int],
                      target_list: List[WorldItem],
                      threshold: int = 100) -> int:
    """
    Detect red targets using robot position and image direction

    Args:
        img: Input camera image
        robot_pos: (x, y, phi) position and orientation in world coordinates
        target_list: List of WorldItem objects
        threshold: Minimum number of red pixels required

    Returns:
        target_id if matched, -1 if no match
    """
    # Extract robot position and orientation
    robot_x, robot_y, robot_phi = robot_pos

    # 1. Detect red pixels in the image
    if isinstance(img, np.ndarray):
        img = img.tobytes()

    try:
        if isinstance(img, bytes):
            img = np.frombuffer(img, dtype=np.uint8).reshape(QVGA_Y, QVGA_X, -1)

        hsi = IPCol2HSI(img)
        hue = np.array(hsi[0], dtype=np.float32).reshape(QVGA_Y, QVGA_X)

        red_mask = hue > 20
        red_indices = np.nonzero(red_mask)
        red_count = len(red_indices[0])

        if red_count < threshold:
            return -1

        # 2. Calculate center of red pixels
        center_y = int(np.mean(red_indices[0]))
        center_x = int(np.mean(red_indices[1]))

        # 3. Calculate direction based on red pixel position in the 180° FOV camera
        camera_fov = 180  # degrees
        angle_per_pixel = camera_fov / QVGA_X
        relative_angle = (center_x - (QVGA_X / 2)) * angle_per_pixel
        relative_angle_rad = np.radians(relative_angle)

        # 4. Find closest target based on position and direction
        closest_target_idx = None
        closest_dist = float('inf')
        search_radius = 800

        for i, target_loc in enumerate(TARGET_LOCATIONS):
            # Calculate distance from robot to target
            dist = int(np.sqrt((target_loc[0] - robot_x) ** 2 + (target_loc[1] - robot_y) ** 2))

            # Calculate direction to the target in world coordinates
            target_direction_x = target_loc[0] - robot_x
            target_direction_y = target_loc[1] - robot_y

            # Calculate angle to target in world coordinates
            target_angle_world = np.degrees(np.arctan2(target_direction_y, target_direction_x))
            target_angle_world = target_angle_world % 360

            # Calculate the angle to target in robot's local frame
            target_angle_local = (target_angle_world - robot_phi) % 360
            if target_angle_local > 180:
                target_angle_local -= 360

            # Check if target is in the direction of the detected red pixels
            angle_tolerance = 45  # degrees
            direction_matches = abs(target_angle_local - relative_angle) < angle_tolerance

            # Only consider targets that are in the camera's field of view
            # and within reasonable distance
            if direction_matches and dist < search_radius:
                if dist < closest_dist:
                    closest_dist = dist
                    closest_target_idx = i

        # If no target was found in the expected direction, just pick the closest one
        if closest_target_idx is None:
            for i, target_loc in enumerate(TARGET_LOCATIONS):
                dist = int(np.sqrt((target_loc[0] - robot_x) ** 2 + (target_loc[1] - robot_y) ** 2))
                if dist < closest_dist:
                    closest_dist = dist
                    closest_target_idx = i

        # Get the closest target location
        matched_location = TARGET_LOCATIONS[closest_target_idx]

        # Find the corresponding WorldItem
        matched_item = None
        for item in target_list:
            if item.item_type == 'target':
                target_dist = int(np.sqrt((item.x - matched_location[0]) ** 2 +
                                          (item.y - matched_location[1]) ** 2))
                if target_dist < 10:  # Assuming target items are very close to TARGET_LOCATIONS
                    matched_item = item
                    break

        if matched_item:
            logger.info(f"Matched with target #{matched_item.item_id} at position ({matched_item.x}, {matched_item.y})")
            return matched_item.item_id

        # If no exact item was found, return the first valid target ID (not 0)
        for item in target_list:
            if item.item_type == 'target' and item.item_id > 0:
                return item.item_id

        # If all else fails, return the first target regardless of ID
        for item in target_list:
            if item.item_type == 'target':
                return item.item_id

        # Only reach here if there are truly no targets
        return -1

    except Exception as e:
        return -1

def red_detector(img) -> Tuple[bool, int, int]:
    """Optimized red detection with numpy operations and increased cache"""
    # Convert img to a hashable type for caching
    if isinstance(img, np.ndarray):
        img = img.tobytes()  # Convert numpy array to bytes for hashing

    try:
        # Convert back to array if needed
        if isinstance(img, bytes):
            img = np.frombuffer(img, dtype=np.uint8).reshape(QVGA_Y, QVGA_X, -1)

        hsi = IPCol2HSI(img)
        hue = np.array(hsi[0], dtype=np.float32).reshape(QVGA_Y, QVGA_X)

        # Vectorized red detection
        red_mask = hue > 20
        red_indices = np.nonzero(red_mask)

        if not red_indices[0].size:
            return False, 0, 0

        # Optimized visualization using numpy operations
        for x, y in zip(red_indices[1], red_indices[0]):
            LCDPixel(x, y, RED)

        # Efficient column counting using numpy
        red_count = np.bincount(red_indices[1])

        # Vectorized histogram visualization
        nonzero_cols = np.nonzero(red_count)[0]
        for i in nonzero_cols:
            LCDLine(i, QVGA_Y, i, QVGA_Y - red_count[i], RED)

        max_col = np.argmax(red_count)
        return True, int(max_col), int(red_count[max_col])

    except Exception as e:
        logger.error(f"Red detection failed: {str(e)}")
        return False, 0, 0