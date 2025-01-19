from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Position:
    """Immutable robot position data with type hints"""
    x: int
    y: int
    phi: int

    def __hash__(self):
        """Custom hash implementation for floating point values"""
        return hash((self.x, self.y, self.phi))

    def __eq__(self, other):
        """Custom equality check for floating point values"""
        if not isinstance(other, Position):
            return False
        return (self.x == other.x and
                self.y == other.y and
                self.phi == other.phi)

    def __iter__(self):
        """Make position iterable for unpacking"""
        yield self.x
        yield self.y
        yield self.phi

    def __str__(self) -> str:
        """String representation of position"""
        return f"x={self.x}, y={self.y}, phi={self.phi}"

    def to_dict(self) -> Dict[str, int]:
        """Type-safe dictionary conversion"""
        return {"x": self.x, "y": self.y, "phi": self.phi}

    def describe(self) -> str:
        """Generate a natural language description of the position"""
        return f"position ({self.x}, {self.y}) facing {self.phi}°"

@dataclass
class Action:
    """
    Represents a robotic action with position tracking and safety checks.
    Uses slots for memory optimization and frozen=True for immutability where possible.
    """
    action: str
    direction: str = ""
    distance: int = 0
    angle: int = 0
    safe: bool = True
    executed: bool = False
    pos_before: Dict[str, float] = field(default_factory=dict)
    pos_after: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        """Memory-efficient string representation"""
        return (
            f"Action({self.action}, dir={self.direction}, "
            f"dist={self.distance}, angle={self.angle}, safe={self.safe})"
        )

    def __hash__(self):
        # Only hash the immutable attributes
        return hash((self.action, self.direction, self.distance, self.angle))

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.action == other.action and
                self.direction == other.direction and
                self.distance == self.distance and
                self.angle == self.angle)

    def to_dict(self, step: Optional[int] = None, target_lost: bool = False) -> Dict:
        """
        Convert action to dictionary with optional step and target_lost parameters

        Args:
            step: Optional step number
            target_lost: Whether target was lost during action

        Returns:
            Dictionary containing action state
        """
        return {
            "step": step,
            "action": self.action,
            "direction": self.direction,
            "distance": self.distance,
            "angle": self.angle,
            "safe": self.safe,
            "executed": self.executed,
            "pos_before": self.pos_before,
            "pos_after": self.pos_after,
            "target_lost": target_lost
        }

    def from_dict(self, action_dict: Dict) -> None:
        """Safe dictionary update with validation"""
        valid_attrs = {'action', 'direction', 'distance', 'angle'}
        for attr, value in action_dict.items():
            if attr in valid_attrs:
                setattr(self, attr, value)

    @lru_cache(maxsize=256)
    def _calculate_safety(self, distance: int, required_clearance: int = 100) -> bool:
        """Optimized safety calculation with increased cache size"""
        return distance >= abs(self.distance) + required_clearance

    def is_safe(self, scan: List[int], range_degrees: int = 30) -> bool:
        """
        Enhanced safety check for C integer array input
        """
        if not scan:  # Early validation
            self.safe = False
            return False

        offset = 179 if self.direction != "backward" else 0

        # Direct array indexing instead of dictionary get()
        scan_values = [scan[offset + direction]
                       for direction in range(-range_degrees, range_degrees + 1)]

        self.safe = all(self._calculate_safety(distance) for distance in scan_values)
        return self.safe


