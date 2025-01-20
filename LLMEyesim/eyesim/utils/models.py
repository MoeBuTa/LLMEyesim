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
