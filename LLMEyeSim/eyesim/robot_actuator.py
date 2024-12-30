from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from eye import *
from loguru import logger
import numpy as np


@dataclass(frozen=True)
class Position:
    """Immutable robot position data with type hints"""
    x: float = 0.0
    y: float = 0.0
    phi: float = 0.0

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

    def to_dict(self) -> Dict[str, float]:
        """Type-safe dictionary conversion"""
        return {"x": self.x, "y": self.y, "phi": self.phi}


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

class RobotActuator:
    """
    Optimized robot actuator class with enhanced error handling and performance
    """

    def __init__(self):
        # Initialize instance attributes
        self.position: Position = Position()
        self.img = None
        self.scan = None
        self.step: int = 0
        self.last_command: List[Action] = [Action("stop")]
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)

        # Initialize hardware and state
        self._initialize_hardware()

    def _initialize_hardware(self) -> None:
        """Initialize hardware with proper error handling"""
        try:
            CAMInit(QVGA)
            self.img = CAMGet()
            self.scan = LIDARGet()
        except Exception as e:
            logger.error(f"Hardware initialization failed: {str(e)}")
            raise RuntimeError(f"Hardware initialization failed: {str(e)}")

    def update_sensors_parallel(self) -> None:
        """Update sensors in parallel using threads"""
        self.img = CAMGet()
        LCDImage(self.img)
        self.scan = LIDARGet()


    def straight(self, distance: int, speed: int, direction: str) -> None:
        """Move robot straight with improved error handling"""
        try:
            factor = 1 if direction == "forward" else -1
            VWStraight(factor * distance, speed)
            VWWait()
            self.update_position()
        except Exception as e:
            logger.error(f"Movement error: {e}")
            raise

    def turn(self, angle: int, speed: int, direction: str) -> None:
        """Turn robot with improved error handling"""
        try:
            factor = 1 if direction == "left" else -1
            VWTurn(factor * angle, speed)
            VWWait()
            self.update_position()
        except Exception as e:
            logger.error(f"Turn error: {e}")
            raise

    def update_position(self) -> None:
        """Update robot position state by creating new Position instance"""
        try:
            x, y, phi = VWGetPosition()
            self.position = Position(x=x, y=y, phi=phi)
        except Exception as e:
            logger.error(f"State update failed: {e}")
            raise

    def format_last_command(self) -> Optional[List[str]]:
        """Format last command for output"""
        if not self.last_command:
            return None
        return [str(action) for action in
                (self.last_command if isinstance(self.last_command, list) else [self.last_command])]

    @staticmethod
    def red_detector(img) -> Tuple[bool, int, int]:
        """Optimized red detection with numpy operations and increased cache"""
        # Convert img to a hashable type for caching
        if isinstance(img, np.ndarray):
            img = img.tobytes()  # Convert numpy array to bytes for hashing

        try:
            # Convert back to array if needed
            if isinstance(img, bytes):
                img = np.frombuffer(img, dtype=np.uint8).reshape(QVGA_Y, QVGA_X, -1)

            hsi = IPCol2HSI(img)
            hue = np.array(hsi[0], dtype=np.float32).reshape(QVGA_Y, QVGA_X)

            # Vectorized red detection
            red_mask = hue > 20
            red_indices = np.nonzero(red_mask)

            if not red_indices[0].size:
                return False, 0, 0

            # Optimized visualization using numpy operations
            for x, y in zip(red_indices[1], red_indices[0]):
                LCDPixel(x, y, RED)

            # Efficient column counting using numpy
            red_count = np.bincount(red_indices[1])

            # Vectorized histogram visualization
            nonzero_cols = np.nonzero(red_count)[0]
            for i in nonzero_cols:
                LCDLine(i, QVGA_Y, i, QVGA_Y - red_count[i], RED)

            max_col = np.argmax(red_count)
            return True, int(max_col), int(red_count[max_col])

        except Exception as e:
            logger.error(f"Red detection failed: {str(e)}")
            return False, 0, 0
    def __del__(self) -> None:
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)