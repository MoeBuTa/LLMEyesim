from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Tuple, TypedDict, Set, Dict

from LLMEyesim.eyesim.utils.config import DIRECTION_RANGES


@dataclass
class PlotConfig:
    """Configuration for plot styling and parameters"""
    figsize: Tuple[int, int] = (3, 3)
    dpi: int = 100
    style: Literal["white", "darkgrid", "dark", "whitegrid", "ticks"] = "white"  # Changed default to "white"
    context: str = "notebook"  # Added context parameter
    cmap: str = "viridis"
    alpha: float = 0.7
    grid_color: str = "gray"
    grid_alpha: float = 0.3
    marker_size: int = 10

@dataclass
class TaskPaths:
    task_name: str
    img_path: Path
    task_path: Path
    state_path: Path
    llm_reasoning_record_path: Path
    llm_action_record_path: Path


@dataclass(frozen=True)
class ObjectPosition:
    """Position and characteristics of a detected object"""
    item_id: int
    item_name: str
    item_type: str
    distance: int
    angle: int
    lidar_distance: int
    confidence: float
    x: int
    y: int

    def __str__(self) -> str:
        return f"Found a {self.item_type}: {self.item_name}, object id: {self.item_id} at ({self.x}, {self.y})\n"

    def describe(self) -> str:
        """Generate a natural language description of the object"""
        # Determine distance description
        if self.distance < 500:
            distance_desc = "very close"
        elif self.distance < 1000:
            distance_desc = "nearby"
        elif self.distance < 2000:
            distance_desc = "at a moderate distance"
        else:
            distance_desc = "far away"

        # Determine direction
        direction = self.get_direction()
        direction_desc = f"to the {direction}" if direction else "in an undefined direction"

        # Format confidence level
        confidence_desc = self._format_confidence()

        # Combine descriptions
        description = (
            f"Detected a {self.item_type} ({self.item_name}) {distance_desc} {direction_desc} "
            f"with {confidence_desc} confidence. "
            f"It is located at coordinates ({self.x}, {self.y}), "
            f"{self.distance} units away at angle {self.angle}°."
        )

        # Add lidar comparison if relevant
        if abs(self.lidar_distance - self.distance) > 50:
            description += (
                f" Note: Lidar reading ({self.lidar_distance} units) "
                f"differs from calculated distance."
            )

        return description

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to be between 0 and 360 degrees"""
        return angle % 360

    def get_direction(self) -> str:
        """Determine the cardinal or intercardinal direction of the object"""
        normalized_angle = self._normalize_angle(self.angle)

        for direction, (start, end) in DIRECTION_RANGES.items():
            if start <= end:
                if start <= normalized_angle <= end:
                    return direction
            else:  # Handle wrap-around case (e.g., east: 337.5-22.5)
                if normalized_angle >= start or normalized_angle <= end:
                    return direction
        return ""

    def _format_confidence(self) -> str:
        """Format confidence level for natural language description"""
        if self.confidence >= 0.9:
            return "very high"
        elif self.confidence >= 0.7:
            return "high"
        elif self.confidence >= 0.5:
            return "moderate"
        elif self.confidence >= 0.3:
            return "low"
        else:
            return "very low"

    def get_technical_details(self) -> Dict:
        """Get technical details for debugging or analysis"""
        return {
            "id": self.item_id,
            "name": self.item_name,
            "type": self.item_type,
            "position": {"x": self.x, "y": self.y},
            "polar": {"distance": self.distance, "angle": self.angle},
            "lidar_distance": self.lidar_distance,
            "confidence": self.confidence,
            "direction": self.get_direction()
        }


@dataclass(frozen=True)
class ObstacleRegion:
    start_angle: int
    end_angle: int
    min_distance: int
    angular_width: int

    def __str__(self) -> str:
        directions = self.get_directions()
        direction_str = ", ".join(directions) if directions else "no specific direction"
        return f"Obstacle Region from degree {self.start_angle} to degree {self.end_angle} " \
               f"with minimum distance {self.min_distance} in direction(s): {direction_str}\n"

    def describe(self) -> str:
        """
        Generate a natural language description of the obstacle region.
        Returns:
            str: A human-readable description of the obstacle's location and characteristics
        """
        directions = self.get_directions()

        # Determine distance description
        if self.min_distance < 500:
            distance_desc = "very close"
        elif self.min_distance < 1000:
            distance_desc = "nearby"
        elif self.min_distance < 2000:
            distance_desc = "at a moderate distance"
        else:
            distance_desc = "far away"

        # Determine size description based on angular width
        if self.angular_width < 10:
            size_desc = "narrow"
        elif self.angular_width < 30:
            size_desc = "moderate-sized"
        elif self.angular_width < 60:
            size_desc = "wide"
        else:
            size_desc = "very wide"

        # Format direction description
        if not directions:
            direction_desc = "in an undefined direction"
        elif len(directions) == 1:
            direction_desc = f"to the {next(iter(directions))}"
        elif len(directions) == 2:
            direction_desc = f"spanning from {' to '.join(directions)}"
        else:
            direction_list = list(directions)
            direction_desc = f"spanning multiple directions ({', '.join(direction_list[:-1])} and {direction_list[-1]})"

        # Combine all descriptions
        description = f"Detected a {size_desc} obstacle {distance_desc} {direction_desc}. "

        # Add specific details
        description += f"It spans {self.angular_width} degrees (from {self.start_angle}° to {self.end_angle}°) "
        description += f"and is approximately {self.min_distance} units away at its closest point."

        return description

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to be between 0 and 360 degrees"""
        return angle % 360

    def _angle_in_range(self, angle: float, start: float, end: float) -> bool:
        """
        Check if an angle falls within a range, handling wrap-around cases.
        Args:
            angle: The angle to check
            start: Start of the range
            end: End of the range
        Returns:
            bool: True if angle is in range, False otherwise
        """
        angle = self._normalize_angle(angle)
        start = self._normalize_angle(start)
        end = self._normalize_angle(end)

        if start <= end:
            return start <= angle <= end
        else:  # Range wraps around 360
            return angle >= start or angle <= end

    def get_directions(self) -> Set[str]:
        """
        Determine which directions this obstacle region falls into.
        Returns:
            Set of direction strings that this obstacle intersects with
        """
        directions = set()

        # For each direction range, check if our obstacle region overlaps
        for direction, (range_start, range_end) in DIRECTION_RANGES.items():
            # Check if any part of the obstacle region falls within this direction range
            # We need to check both start and end angles, plus whether the region entirely contains the direction

            # Normalize our angles
            start = self._normalize_angle(self.start_angle)
            end = self._normalize_angle(self.end_angle)

            # Case 1: Start angle falls in range
            if self._angle_in_range(start, range_start, range_end):
                directions.add(direction)
                continue

            # Case 2: End angle falls in range
            if self._angle_in_range(end, range_start, range_end):
                directions.add(direction)
                continue

            # Case 3: Range is completely contained within obstacle region
            if self._angle_in_range(range_start, start, end):
                directions.add(direction)
                continue

        return directions

