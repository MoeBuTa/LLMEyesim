from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Tuple, TypedDict


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
class ObstacleRegion:
    start_angle: int
    end_angle: int
    min_distance: int
    angular_width: int

    def __str__(self) -> str:
        return f"Obstacle Region from degree {self.start_angle} to degree {self.end_angle} with minimum distance {self.min_distance}\n"

@dataclass(frozen=True)
class ObjectPosition:
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
