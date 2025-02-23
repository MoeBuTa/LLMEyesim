import math
from typing import List, Tuple

from LLMEyesim.eyesim.generator.models import WorldItem
from loguru import logger


def calculate_object_positions(
        robot_pos: Tuple[int, int],  # x, y
        objects: List[WorldItem],
        lidar_data: List[int],
        distance_threshold: int = 200
) -> List[WorldItem]:
    """
    Match objects with lidar readings in 90-270 degree range.

    Args:
        robot_pos: Tuple of (x, y) representing robot position
        objects: List of WorldItem objects containing object information
        lidar_data: List of 360 integer distance readings (index 0 = 0 degrees)
        distance_threshold: Maximum distance difference to consider a match

    Returns:
        List of ObjectPosition objects containing detected objects
    """

    detected_objects = []
    robot_x, robot_y = robot_pos

    # Only process lidar data from 90 to 270 degrees
    for angle in range(90, 271):
        lidar_distance = lidar_data[angle]

        # Check each object against current lidar reading
        for obj in objects:
            # Calculate actual distance to object
            dx = obj.x - robot_x
            dy = obj.y - robot_y
            actual_distance = int((dx * dx + dy * dy) ** 0.5)

            # If distance matches lidar reading within threshold
            if abs(actual_distance - lidar_distance) <= distance_threshold and actual_distance < 2000:
                detected_objects.append(obj)
                break  # Move to next angle once we find a matching object

    return detected_objects


def update_object_positions(
        new_detected_objects: List[WorldItem],
        detected_objects: List[WorldItem],
) -> List[WorldItem]:
    """
    Update object positions based on new detections and previous detections.
    Now with logging for better debugging.
    """

    # Update existing detections or add new ones
    for new_obj in new_detected_objects:
        flag = True
        for obj in detected_objects:
            if new_obj.item_id != obj.item_id:
                continue
            flag = False
            break
        if flag:
            logger.info(f"Adding new object {new_obj.item_id} {new_obj.item_name} at ({new_obj.x}, {new_obj.y})")
            detected_objects.append(new_obj)
    return detected_objects


def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Calculate the distance between two points.

    Args:
        x1 (int): x coordinate of the first point
        y1 (int): y coordinate of the first point
        x2 (int): x coordinate of the second point
        y2 (int): y coordinate of the second point
    Returns:
        int: The distance between the two points
    """
    return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def is_movement_safe(
        lidar_data: List[int],
        safety_margin: int = 200  # Additional safety buffer (units)
) -> bool:
    """
    Check if moving in a specific direction for a given distance is safe.

    Args:
        lidar_data: 360-degree lidar readings (integer values)
        safety_margin: Additional safety buffer distance

    Returns:
        bool: - is_safe: True if movement is safe, False otherwise

    LIDAR: 0 ~ 359
    180 is the front of the robot
    0 is the back of the robot
    """

    # Calculate the range of angles to check
    start_angle = 150
    end_angle = 210

    # Check each angle in the range
    min_distance = 6000
    min_distance_angle = None
    # Log distances for each angle being checked
    for angle in range(start_angle, end_angle + 1):
        distance = lidar_data[angle]
        if distance < min_distance:
            min_distance = distance
            min_distance_angle = angle

    # Check if the path is clear
    if min_distance <= safety_margin:
        logger.warning(
            f"Movement is unsafe! Min distance {min_distance} at angle {min_distance_angle}")
        return False
    return True
