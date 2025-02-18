from dataclasses import dataclass
from typing import List, Optional, Dict

from LLMEyesim.eyesim.actuator.models import Position
from LLMEyesim.eyesim.utils.models import ObjectPosition


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
    execution_status: bool = False  # False for not executed, True for executed
    detail: Optional[str] = None  # Additional detail about execution/non-execution

    def __str__(self) -> str:
        return f"Move {self.direction} by {self.distance}"

    def describe(self) -> str:
        """Generate a natural language description of the action"""
        return f"moved {self.distance} units {self.direction}"

    def get_execution_description(self) -> str:
        """Generate a natural language description based on execution status"""
        action_desc = f"{self.distance} units {self.direction}"
        if self.execution_status:
            base_desc = f"has moved {action_desc}"
        else:
            base_desc = f"will move {action_desc}"

        # Add detail if available
        if self.detail:
            return f"{base_desc} ({self.detail})"
        return base_desc


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

        # Describe pending actions
        if self.action_queue:
            description.append("\nPending Actions:")
            for i, action in enumerate(self.action_queue, start=1):
                next_step = len(self.executed_actions) + i
                description.append(f"Step {next_step}: Robot {action.get_execution_description()}")
        else:
            description.append("\nNo pending actions in the queue.")

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


@dataclass(frozen=True)
class ExplorationRecord:
    """Record of the exploration state at a particular step"""
    object_positions: List[ObjectPosition]
    reached_targets: List[int]
    scan_data: List[int]
    step: int

    def __str__(self) -> str:
        """Technical string representation of the exploration record"""
        parts = [f"Exploration Record - Step {self.step}"]

        # Add objects
        if self.object_positions:
            parts.extend([str(pos) for pos in self.object_positions])

        # Add targets
        if self.reached_targets:
            parts.append(f"Reached targets: {', '.join(map(str, self.reached_targets))}")

        return "\n".join(parts)

    def describe(self) -> str:
        """Generate a natural language description of the exploration state"""
        # Describe step
        descriptions: List = [f"At exploration step {self.step}:"]

        # Describe objects by type
        if self.object_positions:
            # Group objects by type
            objects_by_type: Dict[str, List[ObjectPosition]] = {}
            for obj in self.object_positions:
                if obj.item_type not in objects_by_type:
                    objects_by_type[obj.item_type] = []
                objects_by_type[obj.item_type].append(obj)

            # Describe each type
            for item_type, objects in objects_by_type.items():
                if len(objects) == 1:
                    descriptions.append(f"Found one {item_type}: {objects[0].describe()}")
                else:
                    descriptions.append(f"Found {len(objects)} {item_type}s:")
                    descriptions.extend([f"- {obj.describe()}" for obj in objects])
        else:
            descriptions.append("No objects detected in this step.")

        # Describe reached targets
        if self.reached_targets:
            targets_str = ", ".join(map(str, self.reached_targets[:-1]))
            if targets_str:
                targets_str += " and "
            targets_str += str(self.reached_targets[-1])
            descriptions.append(
                f"Successfully reached target{'' if len(self.reached_targets) == 1 else 's'} {targets_str}."
            )

        # Describe LiDAR scan data
        if self.scan_data:
            descriptions.append(f"LiDAR scan data: {self.scan_data}")

        return "\n".join(descriptions)

    def get_objects_by_type(self, item_type: str) -> List[ObjectPosition]:
        """Get all objects of a specific type"""
        return [obj for obj in self.object_positions if obj.item_type == item_type]

    def get_nearest_object(self, x: int, y: int) -> Optional[ObjectPosition]:
        """Find the nearest object to a given position"""
        if not self.object_positions:
            return None
        return min(
            self.object_positions,
            key=lambda obj: ((obj.x - x) ** 2 + (obj.y - y) ** 2) ** 0.5
        )

    def get_objects_in_direction(self, direction: str) -> List[ObjectPosition]:
        """Get all objects in a specific direction"""
        return [
            obj for obj in self.object_positions
            if obj.get_direction() == direction
        ]

@dataclass(frozen=True)
class ExplorationRecordList:
    """List of exploration records with analysis capabilities"""
    records: List[ExplorationRecord]

    def __str__(self) -> str:
        """Technical string representation of the exploration history"""
        if not self.records:
            return "No exploration records available."

        latest = self.records[-1]
        parts = [
            "Current Exploration Status:",
            *[str(pos) for pos in latest.object_positions]
        ]

        if latest.reached_targets:
            parts.append(
                f"Reached targets: {[f'target id: {target}' for target in latest.reached_targets]}"
            )

        if latest.scan_data:
            parts.append(f"LiDAR scan data: {[i for i in latest.scan_data]}")

        return "\n".join(parts)

    def describe_history(self, start_step: Optional[int] = None, end_step: Optional[int] = None) -> str:
        """
        Generate a natural language description of the exploration history
        Args:
            start_step: Optional starting step (None for beginning)
            end_step: Optional ending step (None for latest)
        """
        relevant_records = self.records
        if start_step is not None:
            relevant_records = [r for r in relevant_records if r.step >= start_step]
        if end_step is not None:
            relevant_records = [r for r in relevant_records if r.step <= end_step]

        if not relevant_records:
            return "No exploration records available for the specified step range."

        descriptions = []
        for record in relevant_records:
            descriptions.append(record.describe())
            descriptions.append("-" * 40)  # Separator

        return "\n".join(descriptions)

    def get_record_at_step(self, step: int) -> Optional[ExplorationRecord]:
        """Get the exploration record for a specific step"""
        for record in self.records:
            if record.step == step:
                return record
        return None

    def get_latest_record(self) -> Optional[ExplorationRecord]:
        """Get the most recent exploration record"""
        return self.records[-1] if self.records else None

    def analyze_changes(self, step1: int, step2: int) -> str:
        """Analyze changes between two steps"""
        record1 = self.get_record_at_step(step1)
        record2 = self.get_record_at_step(step2)

        if not record1 or not record2:
            return f"Unable to analyze changes: missing records for steps {step1} and/or {step2}"

        changes = []

        # Analyze object changes
        new_objects = set(record2.object_positions) - set(record1.object_positions)
        lost_objects = set(record1.object_positions) - set(record2.object_positions)

        if new_objects:
            changes.append("New objects detected:")
            changes.extend([f"- {obj.describe()}" for obj in new_objects])

        if lost_objects:
            changes.append("Objects no longer visible:")
            changes.extend([f"- {obj.describe()}" for obj in lost_objects])

        # Analyze target changes
        new_targets = set(record2.reached_targets) - set(record1.reached_targets)
        if new_targets:
            targets_str = ", ".join(map(str, new_targets))
            changes.append(f"Newly reached targets: {targets_str}")

        if not changes:
            return f"No significant changes detected between steps {step1} and {step2}"

        return "\n".join(changes)

    def get_exploration_summary(self) -> str:
        """Generate a summary of the entire exploration"""
        if not self.records:
            return "No exploration records available."

        first_record = self.records[0]
        latest_record = self.records[-1]

        total_objects_found = len({
            obj.item_id for record in self.records
            for obj in record.object_positions
        })


        summary = [
            f"Exploration Summary ({len(self.records)} steps):",
            f"- Started at step {first_record.step}",
            f"- Current step: {latest_record.step}",
            f"- Total unique objects found: {total_objects_found}",
            f"- Total targets reached: {len(latest_record.reached_targets)}",
        ]

        return "\n".join(summary)


"""
Example Usage:
# Regular description:
At exploration step 1:
Found one target: Target A located at coordinates (566, 566)...

# Change analysis:
New objects detected:
- Landmark B located at coordinates (0, 1200)...
New obstacles detected:
- A moderate-sized obstacle at moderate distance...

# Summary:
Exploration Summary (2 steps):
- Started at step 0
- Current step: 1
- Total unique objects found: 2
- Total targets reached: 1
"""