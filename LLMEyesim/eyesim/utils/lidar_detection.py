import math
from typing import List, Tuple

from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.eyesim.utils.models import ObjectPosition
from loguru import logger


def normalize_angle(angle: int) -> int:
    """Normalize angle to [0, 360) degrees"""
    return angle % 360


def angle_in_fov(angle: int, robot_phi: int) -> bool:
    """
    Check if an angle is within the camera's 180-degree FOV
    considering robot's orientation

    Args:
        angle: Global angle to check (in degrees)
        robot_phi: Robot's orientation (in degrees)
    """
    # Normalize both angles
    norm_angle = normalize_angle(angle)
    norm_phi = normalize_angle(robot_phi)

    # Calculate relative angle to robot's orientation
    rel_angle = normalize_angle(norm_angle - norm_phi)

    # Check if angle is within FOV [90,270] relative to robot orientation
    return 90 <= rel_angle <= 270


def calculate_object_positions(
        robot_pos: Tuple[int, int, int],  # x, y, phi
        objects: List['WorldItem'],
        lidar_data: List[int],
        distance_threshold: int = 200
) -> List[ObjectPosition]:
    """
    Match lidar readings with known objects in the environment, considering camera FOV.

    Args:
        robot_pos: Tuple of (x, y, phi) representing robot position and orientation (phi in degrees)
        objects: List of WorldItem objects containing object information
        lidar_data: List of 360 integer distance readings (index 0 = 0 degrees, 359 = 359 degrees)
        distance_threshold: Maximum distance difference in meters to consider a match

    Returns:
        List of ObjectPosition objects containing detected objects with their properties
    """
    detected_objects: List[ObjectPosition] = []
    x, y, phi = robot_pos

    # Process each object
    for obj in objects:
        try:
            # Calculate relative position in global coordinates
            dx = obj.x - x
            dy = obj.y - y

            # Calculate distance to object
            distance = int((dx * dx + dy * dy) ** 0.5)

            # Calculate global angle to object (in degrees)
            global_angle = int(math.degrees(math.atan2(dy, dx)))
            global_angle = normalize_angle(global_angle)

            # Calculate angle relative to robot's orientation
            relative_angle = normalize_angle(global_angle - phi)
            # Check if object is within camera FOV
            if not angle_in_fov(global_angle, phi):
                continue

            # Find closest lidar reading index for the relative angle
            lidar_idx = int(relative_angle) % 360
            lidar_distance = lidar_data[lidar_idx]

            # Check if lidar distance matches object distance within thresholds
            if abs(distance - lidar_distance) <= distance_threshold:
                # Use a fixed scanning window (5 degrees on each side)
                scan_window = 5
                angle_matches = 0
                consecutive_matches = 0
                max_consecutive = 0

                # Scan neighboring angles
                for offset in range(-scan_window, scan_window + 1):
                    check_idx = (lidar_idx + offset) % 360

                    # Skip if outside camera FOV
                    check_angle = normalize_angle(relative_angle + offset)
                    if not angle_in_fov(check_angle, phi):
                        continue

                    if abs(lidar_data[check_idx] - distance) <= distance_threshold:
                        angle_matches += 1
                        consecutive_matches += 1
                        max_consecutive = max(max_consecutive, consecutive_matches)
                    else:
                        consecutive_matches = 0

                # Calculate confidence based on both total matches and consecutive matches
                valid_window_size = sum(1 for offset in range(-scan_window, scan_window + 1)
                                        if angle_in_fov(normalize_angle(relative_angle + offset), phi))

                match_ratio = angle_matches / valid_window_size if valid_window_size > 0 else 0
                consecutive_ratio = max_consecutive / valid_window_size if valid_window_size > 0 else 0
                confidence = round((match_ratio + consecutive_ratio) / 2, 2)

                logger.info(f"Object {obj.item_id} {obj.item_name} detected: {obj.x}, {obj.y}, ")

                detected_objects.append(ObjectPosition(
                    item_id=obj.item_id,
                    item_name=obj.item_name,
                    item_type=obj.item_type,
                    distance=distance,
                    angle=int(relative_angle),
                    lidar_distance=lidar_distance,
                    confidence=confidence,
                    x=obj.x,
                    y=obj.y
                ))
        except (AttributeError, TypeError, ZeroDivisionError) as e:
            logger.error(f"Error processing object: {obj}. Error: {e}")
            continue

    # Sort by confidence
    detected_objects.sort(key=lambda x: x.confidence, reverse=True)

    logger.info(f"Detection complete: found {len(detected_objects)} objects")
    return detected_objects


def update_object_positions(
        new_detected_objects: List[ObjectPosition],
        detected_objects: List[ObjectPosition],
) -> List[ObjectPosition]:
    """
    Update object positions based on new detections and previous detections.
    Now with logging for better debugging.
    """
    logger.info(f"Updating positions: {len(new_detected_objects)} new detections, "
                f"{len(detected_objects)} existing detections")

    # Update existing detections or add new ones
    for new_obj in new_detected_objects:
        # Find if we already have this object
        existing_obj = next((obj for obj in detected_objects
                             if obj.item_id == new_obj.item_id), None)

        if existing_obj is not None:
            # Object exists - check if position changed
            if existing_obj.x != new_obj.x or existing_obj.y != new_obj.y:
                logger.info(f"Updating position for object {new_obj.item_id}: "
                            f"({existing_obj.x}, {existing_obj.y}) -> ({new_obj.x}, {new_obj.y})")
                # Remove old position and add new one
                detected_objects = [obj for obj in detected_objects
                                    if obj.item_id != new_obj.item_id]
                detected_objects.append(new_obj)
        else:
            # New object - add to list
            logger.info(f"Adding new object {new_obj.item_id} at ({new_obj.x}, {new_obj.y})")
            detected_objects.append(new_obj)

    logger.info(f"Update complete: {len(detected_objects)} total objects")
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
    logger.info(f"Checking angles: {start_angle} to {end_angle}")
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