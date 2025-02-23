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



