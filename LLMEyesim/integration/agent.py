import math
from dataclasses import dataclass
from typing import List, Union, Tuple

import numpy as np
from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.eyesim.actuator.config import GRID_DIRECTION
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.eyesim.utils.lidar_detection import (
    calculate_object_positions,
    update_object_positions, is_movement_safe, calculate_distance,
)
from LLMEyesim.eyesim.utils.models import ObjectPosition
from LLMEyesim.eyesim.utils.target_detection import detect_red_target
from LLMEyesim.integration.config import MAXIMUM_STEP
from LLMEyesim.integration.models import (
    ExplorationRecord,
    ExplorationRecordList,
    LLMRecord,
    RobotAction,
    RobotStateRecord,
)
from LLMEyesim.llm.agents.agent import ExecutiveAgent
from LLMEyesim.utils.constants import LOG_DIR
from datetime import datetime

from eye import *

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
logger.add(f"{LOG_DIR}/{formatted_time}.txt", rotation="10 MB", format="{time} | {level} | {message}")


class EmbodiedAgent:
    def __init__(self, agent: ExecutiveAgent, actuator: RobotActuator, world_items: List[WorldItem], **kwargs):
        self.world_items = [item for item in world_items if item.item_id != actuator.robot_id]
        self.agent = agent
        self.actuator = actuator

        self.llm_records: List[LLMRecord] = []

        self.robot_state_record = RobotStateRecord(positions=[], executed_actions=[], action_queue=[])

        # accumulated object positions records
        self.object_detected: List[ObjectPosition] = []

        # exploration records by step
        self.exploration_records: ExplorationRecordList = ExplorationRecordList(records=[])
        self.step = 0

        # search and rescue mission
        self.reached_targets = []
        self.identified_targets = []
        self.target_list = [item for item in world_items if item.item_type == "target"]
        self.target_remaining = len(self.target_list)

        logger.success(
            f"Embodied agent initialized with {self.agent.llm_name}-{self.actuator.robot_name}-{self.actuator.robot_id}-{self.target_list}")

    def _update_records(self, **kwargs):
        if "new_llm_record" in kwargs:
            logger.info(f"Updating LLM record at step {self.step}")
            self.llm_records.append(kwargs["new_llm_record"])
        if "new_exploration_record" in kwargs:
            logger.info(f"Updating exploration record at step {self.step}")
            self.exploration_records.records.append(kwargs["new_exploration_record"])

    def _process_sensors(self) -> Tuple[Union[List[int], None], Union[np.ndarray, None], int, int, int]:
        # process sensors and get environment information
        scan, img = self.actuator.update_sensors()
        x, y, phi = self.actuator.update_position()

        # check if target is detected
        target_id = detect_red_target(img=img, robot_pos=(x, y, phi), target_list=self.target_list)
        if target_id not in self.identified_targets:
            self.identified_targets.append(target_id)

        # update object positions in memory
        current_position = self.robot_state_record.positions[-1]
        new_object_detected = calculate_object_positions(
            robot_pos=(current_position.x, current_position.y, current_position.phi),
            objects=self.world_items, lidar_data=scan)
        self.object_detected = update_object_positions(new_object_detected, self.object_detected)

        # exploration record
        new_exploration_record = ExplorationRecord(object_positions=self.object_detected,
                                                   reached_targets=self.reached_targets,
                                                   scan_data=scan, step=self.step)
        self._update_records(new_exploration_record=new_exploration_record)
        return scan, img, x, y, phi

    def _process_agent(self):
        logger.info(f"Processing agent at step {self.step}")
        response = self.agent.process_v2(exploration_records=str(self.exploration_records),
                                         robot_state=str(self.robot_state_record))
        action_data = response.get('response').get('action_queue', [])

        for action in action_data:
            self.robot_state_record.action_queue.append(
                RobotAction(direction=action['direction'], distance=action['distance']))

        llm_record = LLMRecord(model=response.get('model'), input=response.get('input'), status=response.get('status'),
                               response=response.get('response'), usage=response.get('usage'), step=self.step)
        self._update_records(new_llm_record=llm_record)

    def _check_search_mission_status(self) -> bool:
        """
        Check if the mission is complete.
        """
        if self.target_remaining == 0:
            return True

        pos = self.actuator.position
        for target in self.target_list:
            # consider target as reached if it is detected and within 100mm distance
            for obj in self.object_detected:
                if obj.item_id == target.item_id and calculate_distance(pos.x, pos.y, obj.x,
                                                                        obj.y) < 100:
                    self.target_remaining -= 1
                    self.reached_targets.append(target.item_id)
        return False

    def move_grid(self, distance: int, direction: str) -> None:
        """
        Move the robot in a grid pattern in the specified direction with position checking.
        First turns to target direction, then moves straight while checking position.

        Args:
            distance (int): Distance to move in millimeters
            direction (str): Target direction to move ('N', 'S', 'E', 'W', etc.)
        Raises:
            ValueError: If direction is not one of the valid GRID_DIRECTION values
        """
        if direction not in GRID_DIRECTION:
            raise ValueError(f"Invalid direction: {direction}")

        logger.info(f"{self.actuator.robot_name} {self.actuator.robot_id}: Moving {direction}")
        x, y, phi = self.actuator.position
        target_degree = GRID_DIRECTION[direction]
        self.grid_turn(phi, target_degree)
        logger.info(f"{self.actuator.robot_name} {self.actuator.robot_id}: Moving {distance} mm")
        target_radian = math.radians(phi)
        target_x = x + int(distance * math.cos(target_radian))
        target_y = y + int(distance * math.sin(target_radian))
        self.grid_straight(target_x, target_y)

    def grid_turn(
            self,
            phi: int,
            target_degree: int,
            angle_deviation: int = 5
    ) -> None:
        """
        Calculate and execute the shortest turn between two angles.
        if found target

        Args:
            phi (int): The starting angle in degrees
            target_degree (int): The target angle in degrees
            angle_deviation (int, optional): Acceptable angle deviation in degrees. Defaults to 5.
        Returns:
            None
        """
        diff = ((target_degree - phi) % 360)
        degree_to_turn = diff if diff <= 180 else abs(diff - 360)
        while degree_to_turn > 0:
            VWTurn(5, 100)
            VWWait()
            _, _, _, _, phi = self._process_sensors()
            diff = ((target_degree - phi) % 360)
            degree_to_turn = diff if diff <= 180 else diff - 360

    def grid_straight(self, target_x: int, target_y: int, pos_deviation: int = 10):
        """
        Move the robot straight in a grid pattern.

        Args:
            target_x (int): Target x coordinate in mm
            target_y (int): Target y coordinate in mm
            pos_deviation (int, optional): Acceptable position deviation in mm. Defaults to 10.
        Returns:
            None
        """
        distance = calculate_distance(self.actuator.position.x, self.actuator.position.y, target_x, target_y)
        while distance >= pos_deviation and is_movement_safe(self.actuator.scan):
            VWStraight(10, 100)
            VWWait()
            scan, img, x, y, phi = self._process_sensors()
            distance = calculate_distance(x, y, target_x, target_y)

    def run_agent(self):
        """
        Run the embodied agent with the given actuator.
        """
        # Search and rescue mission
        self._process_sensors()
        while self.step < MAXIMUM_STEP and self.target_remaining > 0:
            self.step += 1
            # update and check search mission status
            if self._check_search_mission_status():
                logger.info(f"Mission completed at step {self.step}")
                return

            # if action queue is empty, run process agent
            # TODO: needs more conditions to trigger process agent
            if not self.robot_state_record.action_queue:
                self._process_agent()

            # move robot by popping next action in queue
            next_action = self.robot_state_record.action_queue.pop(0)
            next_action = RobotAction(direction=next_action.direction, distance=next_action.distance)
            self.move_grid(next_action.distance, next_action.direction)
