from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Tuple


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


class CardinalDirection(str, Enum):
    """Enum for cardinal directions to avoid string literals"""
    NORTH = "north"
    NORTHEAST = "northeast"
    EAST = "east"
    SOUTHEAST = "southeast"
    SOUTH = "south"
    SOUTHWEST = "southwest"
    WEST = "west"
    NORTHWEST = "northwest"


@dataclass
class Obstacle:
    """Dataclass for obstacle information"""
    start_angle: int
    end_angle: int
    start_direction: CardinalDirection
    end_direction: CardinalDirection
    avg_distance: int
    angular_width: int


@dataclass
class DetectedObject:
    """Dataclass for detected object information"""
    name: str
    distance: int
    angle: int
    lidar_distance: int
    confidence: int
    x: int
    y: int