from dataclasses import dataclass
from typing import List, Optional, Dict

from LLMEyesim.eyesim.actuator.models import Position
from LLMEyesim.eyesim.generator.models import WorldItem


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

    def describe(self) -> str:
        """Generate a natural language description of the action"""
        return f"moved {self.distance} units {self.direction}"

    def get_execution_description(self) -> str:
        """Generate a natural language description based on execution status"""
        action_desc = f"{self.distance} units {self.direction}"
        return action_desc


@dataclass(frozen=True)
class RobotStateRecord:
    """Record of the robot's state"""
    positions: List[Position]
    executed_actions: List[RobotAction]
    action_queue: List[RobotAction]

    def describe(self) -> str:
        """Generate a step-by-step description of the robot's complete state history"""
        description = []

        # Describe initial state
        if self.positions:
            initial_pos = self.positions[0]
            description.append(
                f"Step 0: Robot initialized at {initial_pos.describe()}."
            )

        # Describe movement history with steps
        for step, (position, action) in enumerate(zip(self.positions[1:], self.executed_actions), 1):
            description.append(
                f"Step {step}: Robot {action.get_execution_description()} to reach {position.describe()}."
            )

        # Add a separator between history and future actions
        description.append("\nCurrent Status:")

        # Describe current position
        if self.positions:
            current_pos = self.positions[-1]
            current_step = len(self.executed_actions)
            description.append(
                f"The robot is at {current_pos.describe()} (Step {current_step})."
            )


        return "\n".join(description)

    def __str__(self) -> str:
        """
        example output:
        Step 0: Robot initialized at position (0, 0) facing 90°.
        Step 1: Robot moved 5 units north to reach position (0, 5) facing 90°.
        Step 2: Robot moved 3 units east to reach position (3, 5) facing 0°.

        Current Status:
        The robot is at position (3, 5) facing 0° (Step 2).

        Pending Actions:
        Step 3: Will move 2 units north
        Step 4: Will move 1 unit west
        Step 5: Will move 3 units south
        :return:
        """
        return self.describe()

