from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class LLMRecord:
    """Record of the LLM's response"""
    messages: str
    response: str
    step: int

@dataclass(frozen=True)
class RobotAction:
    """Record of the robot's action"""
    direction: str
    distance: int

    def __str__(self) -> str:
        return f"Move {self.direction} by {self.distance}"


@dataclass(frozen=True)
class RobotStateRecord:
    """Record of the robot's state"""
    x: int
    y: int
    phi: int
    action_queue: List[RobotAction]
    step: int

    def __str__(self) -> str:
        return f"Robot at ({self.x}, {self.y}) with action queue {[str(action) for action in self.action_queue]}"

    def get_current_position(self) -> str:
        return f"x={self.x}, y={self.y}"


@dataclass(frozen=True)
class ExplorationRecord:
    """Record of the robot's exploration"""
    item_name: str
    item_type: str
    item_position: Tuple[int, int]
    step: int


    def __str__(self) -> str:
        return f"Found {self.item_name} - {self.item_type} at {str(self.item_position)}"
