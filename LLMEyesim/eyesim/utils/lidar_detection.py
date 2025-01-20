from typing import List, Tuple

from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.eyesim.utils.config import DISTANCE_THRESHOLD
from LLMEyesim.eyesim.utils.models import ObjectPosition

def calculate_object_positions(
        robot_pos: Tuple[int, int],
        objects: List[WorldItem],
        lidar_data: List[int],
        distance_threshold: int = 200
) -> List[ObjectPosition]:
    """
    Match lidar readings with known objects in the environment.

    Args:
        robot_pos: Tuple of (x, y) representing robot position
        objects: List of WorldItem objects containing object information
        lidar_data: List of 360 integer distance readings (index 0 = 0 degrees, 359 = 359 degrees)
        distance_threshold: Maximum distance difference in meters to consider a match

    Returns:
        List of ObjectPosition objects containing detected objects with their properties
    """
    detected_objects: List[ObjectPosition] = []

    # Process each object
    for obj in objects:
        try:
            # Calculate relative position
            dx = obj.x - robot_pos[0]
            dy = obj.y - robot_pos[1]

            # Calculate distance to object (using integer math)
            distance = int((dx * dx + dy * dy) ** 0.5)

            # Calculate angle to object (in degrees)
            angle = 0
            if dx != 0 or dy != 0:
                angle = int((180 / 3.14159) * ((dy / (abs(dx) + abs(dy))) if dx > 0
                                               else (2 - dy / (abs(dx) + abs(dy))) if dx < 0
                else (1 if dy > 0 else 3)) * 90)

            # Normalize angle to [0, 360)
            angle = angle % 360

            # Find closest lidar reading index
            lidar_idx = angle % 360
            lidar_distance = lidar_data[lidar_idx]

            # Check if lidar distance matches object distance within thresholds
            if abs(distance - lidar_distance) <= distance_threshold:
                # Use a fixed scanning window (5 degrees on each side)
                scan_window = 5
                angle_matches = 0
                consecutive_matches = 0
                max_consecutive = 0

                for offset in range(-scan_window, scan_window + 1):
                    check_idx = (lidar_idx + offset) % 360
                    if abs(lidar_data[check_idx] - distance) <= distance_threshold:
                        angle_matches += 1
                        consecutive_matches += 1
                        max_consecutive = max(max_consecutive, consecutive_matches)
                    else:
                        consecutive_matches = 0

                # Calculate confidence based on both total matches and consecutive matches
                match_ratio = angle_matches / (2 * scan_window + 1)
                consecutive_ratio = max_consecutive / (2 * scan_window + 1)
                confidence = round((match_ratio + consecutive_ratio) / 2, 2)

                detected_objects.append(ObjectPosition(
                    item_id=obj.item_id,
                    item_name=obj.item_name,
                    item_type=obj.item_type,
                    distance=distance,
                    angle=angle,
                    lidar_distance=lidar_distance,
                    confidence=confidence,
                    x=obj.x,
                    y=obj.y
                ))
        except (AttributeError, TypeError) as e:
            print(f"Error processing object: {obj}. Error: {e}")
            continue

    # Sort by confidence
    detected_objects.sort(key=lambda x: x.confidence, reverse=True)

    return detected_objects


def update_object_positions(
        new_detected_objects: List[ObjectPosition],
        detected_objects: List[ObjectPosition],
) -> List[ObjectPosition]:
    """
    Update object positions based on new detections and previous detections.

    Args:
        new_detected_objects: List of ObjectPosition objects containing newly detected objects
        detected_objects: List of ObjectPosition objects containing previously detected objects
    """
    # Update existing detections or add new ones
    for new_obj in new_detected_objects:
        # Find if we already have this object
        existing_obj = next((obj for obj in detected_objects
                             if obj.item_id == new_obj.item_id), None)

        if existing_obj is not None:
            # Object exists - check if position changed
            if existing_obj.x != new_obj.x or existing_obj.y != new_obj.y:
                # Remove old position and add new one
                detected_objects_list = [obj for obj in detected_objects
                                         if obj.item_id != new_obj.item_id]
                detected_objects_list.append(new_obj)
        else:
            # New object - add to list
            detected_objects.append(new_obj)
    return detected_objects
