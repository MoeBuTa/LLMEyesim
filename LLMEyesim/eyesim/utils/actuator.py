from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Dict, List, Optional, Tuple

from eye import *
from loguru import logger
import numpy as np

from LLMEyesim.eyesim.utils.config import GRID_DIRECTION


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

    def to_dict(self) -> Dict[str, int]:
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

    def __init__(self, robot_id: int, robot_name: str):
        # Initialize instance attributes
        self.robot_id: int = robot_id
        self.robot_name: str = robot_name
        self.position = Position(0, 0, 0)
        self.img = None
        self.scan = None
        self.step: int = 0
        self.last_command: List[Action] = [Action("stop")]

        # Initialize hardware and state
        self._initialize_hardware()

    def _initialize_hardware(self) -> None:
        """Initialize hardware with proper error handling"""
        try:
            CAMInit(QVGA)
            self.img = CAMGet()
            self.scan = LIDARGet()
            self.update_position()
        except Exception as e:
            logger.error(f"Hardware initialization failed: {str(e)}")
            raise RuntimeError(f"Hardware initialization failed: {str(e)}")

    def update_sensors_parallel(self) -> None:
        """Update sensors in parallel using threads"""
        self.img = CAMGet()
        LCDImage(self.img)
        self.scan = LIDARGet()

    def update_position(self) -> None:
        """Update robot position state by creating new Position instance"""
        try:
            pos = [int.from_bytes(x, byteorder='little') for x in SIMGetRobot(self.robot_id)]
            logger.info(f"Updating position: {pos}")
            self.position = Position(x=pos[0], y=pos[1], phi=pos[3])
        except Exception as e:
            logger.error(f"State update failed: {e}")
            raise

    def format_last_command(self) -> Optional[List[str]]:
        """Format last command for output"""
        if not self.last_command:
            return None
        return [str(action) for action in
                (self.last_command if isinstance(self.last_command, list) else [self.last_command])]

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

    def move_grid(self, distance: int, direction: str, angle_deviation: int = 5, pos_deviation: int = 10) -> None:
        """
        Move the robot in a grid pattern in the specified direction with position checking.
        First turns to target direction, then moves straight while checking position.

        Args:
            distance (int): Distance to move in millimeters
            direction (str): Target direction to move ('N', 'S', 'E', 'W', etc.)
            angle_deviation (int, optional): Acceptable angle deviation in degrees. Defaults to 5.
            pos_deviation (int, optional): Acceptable position deviation in mm. Defaults to 10.

        Raises:
            ValueError: If direction is not one of the valid GRID_DIRECTION values
        """
        if direction not in GRID_DIRECTION:
            raise ValueError(f"Invalid direction: {direction}")

        logger.info(f"Moving {direction}")
        x, y, phi = self.position
        target_degree = GRID_DIRECTION[direction]
        while min((phi - target_degree) % 360, (target_degree - phi) % 360) >= angle_deviation:
            self.grid_turn(phi, target_degree)
            self.update_position()
            x, y, phi = self.position

        logger.info(f"Moving {distance} mm")
        target_radian = math.radians(phi)
        target_x = x + int(distance * math.cos(target_radian))
        target_y = y + int(distance * math.sin(target_radian))
        while self.calculate_distance(x, y, target_x, target_y) >= 10:
            distance = self.calculate_distance(x, y, target_x, target_y)
            self.grid_straight(distance)
            self.update_position()
            x, y, phi = self.position

    @staticmethod
    def grid_turn(
            initial_degree: int,
            target_degree: int
    ) -> None:
        """
        Calculate and execute the shortest turn between two angles.

        Args:
            initial_degree (int): The starting angle in degrees
            target_degree (int): The target angle in degrees
        Returns:
            None
        """
        diff = ((target_degree - initial_degree) % 360)
        degree_to_turn = diff if diff <= 180 else diff - 360
        VWTurn(degree_to_turn, 100)
        VWWait()

    @staticmethod
    def grid_straight(distance: int) -> None:
        """
        Move the robot straight in a grid pattern.

        Args:
            distance (int): Distance to move in millimeters
        Returns:
            None
        """
        VWStraight(distance, 100)
        VWWait()

    @staticmethod
    def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        """
        Calculate the distance between two points.

        Args:
            x1 (int): x coordinate of the first point
            y1 (int): y coordinate of the first point
            x2 (int): x coordinate of the second point
            y2 (int): y coordinate of the second point
        Returns:
            float: The distance between the two points
        """
        return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

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
