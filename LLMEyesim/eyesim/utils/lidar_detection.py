import itertools
from typing import List, Tuple

import numpy as np
from numpy._typing import NDArray

from LLMEyesim.eyesim.utils.config import CARDINAL_DIRECTION_LUT
from LLMEyesim.eyesim.utils.models import CardinalDirection, Obstacle, DetectedObject


def get_cardinal_direction(angle: int) -> CardinalDirection:
    """
    Convert angle to cardinal direction using pre-computed lookup table.
    Args:
        angle: angle in degrees [0, 360)
    Returns:
        Cardinal direction
    """
    return CARDINAL_DIRECTION_LUT[angle % 360]


def detect_obstacles(
        lidar_data: List[int],
        distance_threshold: int = 200
) -> List[Obstacle]:
    """
    Detect continuous obstacle regions from lidar data using vectorized operations.
    Args:
        lidar_data: 360-degree lidar readings
        distance_threshold: Maximum distance to consider as obstacle
    Returns:
        List of obstacle regions
    """
    ## TODO: Fix typings
    # Create boolean mask for obstacles
    obstacle_mask = lidar_data < distance_threshold

    # Find transitions between obstacle and non-obstacle regions
    transitions = np.diff(np.concatenate(([obstacle_mask[-1]], obstacle_mask, [obstacle_mask[0]])))
    start_indices = np.where(transitions == 1)[0]
    end_indices = np.where(transitions == -1)[0]

    # Handle case where obstacle wraps around 360 degrees
    if len(start_indices) != len(end_indices):
        if obstacle_mask[0] and obstacle_mask[-1]:
            start_indices = start_indices[1:]
            end_indices = np.append(end_indices, 360)

    obstacles = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        start_angle = start_idx % 360
        end_angle = (end_idx - 1) % 360

        # Calculate region properties efficiently
        if end_angle < start_angle:
            region_data = np.concatenate((lidar_data[start_angle:], lidar_data[:end_angle + 1]))
        else:
            region_data = lidar_data[start_angle:end_angle + 1]

        obstacles.append(Obstacle(
            start_angle=start_angle,
            end_angle=end_angle,
            start_direction=get_cardinal_direction(start_angle),
            end_direction=get_cardinal_direction(end_angle),
            avg_distance=int(np.mean(region_data)),
            angular_width=(end_angle - start_angle) % 360
        ))

    return obstacles


def calculate_object_positions(
        robot_pos: Tuple[int, int],
        objects: List[dict],
        lidar_data: List[int],
        angle_threshold: int = 2,
        distance_threshold: int = 1,
        scan_window: int = 5
) -> List[DetectedObject]:
    """
    Match lidar readings with known objects using vectorized operations.
    """
    ## TODO: Fix typings
    # Pre-calculate all object properties
    object_positions = np.array([(obj['x'], obj['y']) for obj in objects])
    relative_positions = object_positions - np.array(robot_pos)

    # Vectorized distance and angle calculations
    distances = np.sqrt(np.sum(relative_positions ** 2, axis=1)).astype(np.int64)
    angles = np.degrees(np.arctan2(relative_positions[:, 1], relative_positions[:, 0])).astype(np.int64) % 360
    angles_idx = angles

    detected_objects = []

    # Create sliding window indices for all angles at once
    window_indices = np.array([np.arange(-scan_window, scan_window + 1)])
    all_check_indices = (angles_idx[:, np.newaxis] + window_indices) % 360

    # Get corresponding lidar distances for all window positions
    window_lidar_distances = lidar_data[all_check_indices]

    # Calculate distance matches for all objects and window positions at once
    distance_matches = np.abs(window_lidar_distances - distances[:, np.newaxis]) <= distance_threshold

    for i, obj in enumerate(objects):
        if abs(distances[i] - lidar_data[angles_idx[i]]) <= distance_threshold:
            matches = distance_matches[i]
            angle_matches = np.sum(matches)
            consecutive_matches = np.max(np.array([len(list(g)) for k, g in itertools.groupby(matches) if k]))

            match_ratio = angle_matches * 100 // (2 * scan_window + 1)  # Convert to integer percentage
            consecutive_ratio = consecutive_matches * 100 // (2 * scan_window + 1)  # Convert to integer percentage
            confidence = (match_ratio + consecutive_ratio) // 2

            detected_objects.append(DetectedObject(
                name=obj['name'],
                distance=int(distances[i]),
                angle=int(angles[i]),
                lidar_distance=int(lidar_data[angles_idx[i]]),
                confidence=confidence,
                x=int(obj['x']),
                y=int(obj['y'])
            ))

    return sorted(detected_objects, key=lambda x: x.confidence, reverse=True)


def visualize_detection_with_obstacles(
        robot_pos: Tuple[int, int],
        objects: List[dict],
        lidar_data: List[int],
        detected_objects: List[DetectedObject],
        obstacles: List[Obstacle]
) -> None:
    """
    Optimized visualization function using vectorized operations.
    """
    import matplotlib.pyplot as plt

    # Vectorized coordinate calculations
    angles = np.arange(360) * np.pi / 180
    x_lidar = robot_pos[0] + lidar_data * np.cos(angles)
    y_lidar = robot_pos[1] + lidar_data * np.sin(angles)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all elements efficiently using single calls
    ax.scatter(x_lidar, y_lidar, c='gray', s=1, alpha=0.5, label='Lidar')
    ax.plot(robot_pos[0], robot_pos[1], 'ro', label='Robot')

    # Plot objects
    obj_coords = np.array([(obj['x'], obj['y']) for obj in objects])
    ax.plot(obj_coords[:, 0], obj_coords[:, 1], 'bx', label='Objects')

    # Plot obstacles
    for obstacle in obstacles:
        angles = np.linspace(obstacle.start_angle * np.pi / 180,
                             obstacle.end_angle * np.pi / 180, 100)
        x_obstacle = robot_pos[0] + 200 * np.cos(angles)
        y_obstacle = robot_pos[1] + 200 * np.sin(angles)
        ax.fill(x_obstacle, y_obstacle, 'r', alpha=0.2)

    # Plot detected objects
    if detected_objects:
        detected_coords = np.array([(obj.x, obj.y) for obj in detected_objects])
        ax.plot(detected_coords[:, 0], detected_coords[:, 1], 'g*',
                markersize=15, label='Detected Objects')

    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Lidar Detection Visualization')
    plt.show()


if __name__ == "__main__":
    # Example usage with optimized data structures
    example_robot_position = (0, 0)
    example_objects = [
        {'name': 'Box1', 'x': 2, 'y': 2},
        {'name': 'Chair1', 'x': -1, 'y': 3},
        {'name': 'Table1', 'x': 3, 'y': -1}
    ]

    # Generate random lidar data (as integers)
    example_lidar_data = list(np.random.randint(1, 2, 360))

    # Run detection
    example_detected = calculate_object_positions(example_robot_position, example_objects, example_lidar_data)
    example_obstacles = detect_obstacles(example_lidar_data)

    # Print results
    print("\nDetected Objects:")
    for obj in example_detected:
        direction = get_cardinal_direction(obj.angle)
        print(f"Detected {obj.name} at {obj.angle}° ({direction}) "
              f"and {obj.distance}m with {obj.confidence}% confidence")

    print("\nDetected Obstacles:")
    for obstacle in example_obstacles:
        print(f"Obstacle region from {obstacle.start_angle}° ({obstacle.start_direction}) "
              f"to {obstacle.end_angle}° ({obstacle.end_direction})")
        print(f"  Angular width: {obstacle.angular_width}°")
        print(f"  Average distance: {obstacle.avg_distance}m")

    # Visualize
    visualize_detection_with_obstacles(example_robot_position, example_objects, example_lidar_data, example_detected, example_obstacles)
