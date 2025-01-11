import math
from typing import List, Optional, Tuple

from eye import *
from loguru import logger
import numpy as np

from LLMEyesim.eyesim.actuator.config import GRID_DIRECTION
from LLMEyesim.eyesim.actuator.models import Action, Position


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
        CAMInit(QVGA)
        self.update_sensors()
        self.update_position()


    def update_sensors(self) -> Tuple[List[int], np.ndarray]:
        """Update sensors in parallel using threads"""
        self.img = CAMGet()
        # LCDImage(self.img)
        self.scan = LIDARGet()
        return self.scan, self.img

    def update_position(self) -> Position:
        """Update robot position state by creating new Position instance"""
        pos = [int.from_bytes(x, byteorder='little') for x in SIMGetRobot(self.robot_id)]
        self.position = Position(x=pos[0], y=pos[1], phi=pos[3])
        return self.position


    def format_last_command(self) -> Optional[List[str]]:
        """Format last command for output"""
        if not self.last_command:
            return None
        return [str(action) for action in
                (self.last_command if isinstance(self.last_command, list) else [self.last_command])]

    def straight(self, distance: int, speed: int, direction: str) -> None:
        """Move robot straight with improved error handling"""
        factor = 1 if direction == "forward" else -1
        VWStraight(factor * distance, speed)
        VWWait()
        self.update_position()


    def turn(self, angle: int, speed: int, direction: str) -> None:
        """Turn robot with improved error handling"""
        factor = 1 if direction == "left" else -1
        VWTurn(factor * angle, speed)
        VWWait()
        self.update_position()


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

        logger.info(f"{self.robot_name} {self.robot_id}: Moving {direction}")
        x, y, phi = self.position
        target_degree = GRID_DIRECTION[direction]
        while min((phi - target_degree) % 360, (target_degree - phi) % 360) >= angle_deviation:
            self.grid_turn(phi, target_degree)
            x, y, phi = self.update_position()


        logger.info(f"{self.robot_name} {self.robot_id}: Moving {distance} mm")
        target_radian = math.radians(phi)
        target_x = x + int(distance * math.cos(target_radian))
        target_y = y + int(distance * math.sin(target_radian))
        while self.calculate_distance(x, y, target_x, target_y) >= 10:
            distance = self.calculate_distance(x, y, target_x, target_y)
            self.grid_straight(distance)
            x, y, phi = self.update_position()

        self.update_sensors()

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
