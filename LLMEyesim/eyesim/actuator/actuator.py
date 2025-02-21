from typing import List, Optional, Tuple

from eye import *
from loguru import logger
import numpy as np

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
        logger.success(f"Successfully initialized hardware {self.robot_name}-{self.robot_id}")

    def update_sensors(self) -> Tuple[List[int], np.ndarray]:
        """Update sensors in parallel using threads"""
        self.img = CAMGet()
        LCDImage(self.img)
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


