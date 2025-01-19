from dataclasses import dataclass
from typing import List, Tuple

from LLMEyesim.eyesim.utils.models import ObjectPosition, ObstacleRegion


@dataclass(frozen=True)
class LLMRecord:
    """Record of the LLM's response"""
    model: str
    input: str
    status: str
    response: str
    usage: str
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
    """Record of the exploration state"""
    object_positions: List[ObjectPosition]
    obstacle_regions: List[ObstacleRegion]
    reached_targets: List[int]
    step: int


@dataclass(frozen=True)
class ExplorationRecordList:
    """List of exploration records"""
    records: List[ExplorationRecord]

    # TODO: May need to update this method to return the obstacle regions with multiple steps and the corresponding position of the robot
    def __str__(self) -> str:
        target_reached = f"""We've already reached the following targets before: {[f"target id: {target}, " for target in self.records[-1].reached_targets]} """
        if not self.records[-1].reached_targets:
            target_reached = ""
        return f"""
{[str(position) for position in self.records[-1].object_positions]}
{[str(region) for region in self.records[-1].obstacle_regions]}
{target_reached}
"""
