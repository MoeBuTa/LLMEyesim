import math
from dataclasses import dataclass
from typing import List, Union, Tuple, Dict

import numpy as np
from loguru import logger
from openai import NotGiven
from openai.types.chat import completion_create_params

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.eyesim.actuator.config import GRID_DIRECTION
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.eyesim.utils.lidar_detection import (
    calculate_object_positions,
    update_object_positions, is_movement_safe, calculate_distance,
)

from LLMEyesim.eyesim.utils.target_detection import detect_red_target
from LLMEyesim.integration.config import MAXIMUM_STEP
from LLMEyesim.integration.models import (
    LLMRecord,
    RobotAction,
    RobotStateRecord,
)
from LLMEyesim.llm.agents.agent import ExecutiveAgent
from LLMEyesim.llm.response.models import ActionQueue, WayPointList
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
        self.detected_objects: List[WorldItem] = []

        self.step = 0

        # exploration and targets search mission
        self.history_positions: List[Tuple[int, int, int]] = []
        self.reached_targets: List[int] = []
        self.identified_targets: List[int] = []
        self.target_list: List[WorldItem] = [item for item in world_items if item.item_type == "target"]
        self.target_remaining: int = len(self.target_list)

        logger.success(
            f"Embodied agent initialized with {self.agent.llm_name}-{self.actuator.robot_name}-{self.actuator.robot_id}-{self.target_list}")

    def _process_sensors(self) -> Tuple[Union[List[int], None], Union[np.ndarray, None], int, int, int]:
        # process sensors and get environment information
        try:
            scan, img = self.actuator.update_sensors()
            x, y, phi = self.actuator.update_position()

            # check if target is detected
            target_id = detect_red_target(img=img, robot_pos=(x, y, phi), target_list=self.target_list)
            if target_id not in self.identified_targets:
                self.identified_targets.append(target_id)

            # update object positions in memory
            new_object_detected = calculate_object_positions(
                robot_pos=(x, y),
                objects=self.world_items, lidar_data=scan)
            self.detected_objects = update_object_positions(new_object_detected, self.detected_objects)


        except Exception as e:
            logger.error(f"Error processing sensors: {e}")
            raise e
        return scan, img, x, y, phi

    def _process_agent(self, response_format: completion_create_params.ResponseFormat | NotGiven = WayPointList) -> Dict:
        logger.info(f"Processing agent at step {self.step}")
        message = f"""
current position: {str(self.actuator.position)}
history positions: {self.history_positions}
detected objects: {self.detected_objects}
reached targets: {self.reached_targets}
identified targets: {self.identified_targets}
number of targets remaining: {self.target_remaining}
"""
        response = self.agent.process_v2(message=message, response_format=response_format)
        llm_record = LLMRecord(model=response.get('model'), input=response.get('input'),
                               status=response.get('status'),
                               response=response.get('response'), usage=response.get('usage'), step=self.step)
        self.llm_records.append(llm_record)
        return response

    def _check_search_mission_status(self) -> bool:
        """
        Check if the mission is complete.
        """
        if self.target_remaining == 0:
            return True

        pos = self.actuator.position
        for target in self.target_list:
            # consider target as reached if it is detected and within 100mm distance
            for target_id in self.identified_targets:
                if target_id == target.item_id and calculate_distance(pos.x, pos.y, target.x,
                                                                      target.y) < 100:
                    self.target_remaining -= 1
                    self.reached_targets.append(target.item_id)
        return False

    def move_to_target(self, target_x: int, target_y: int) -> None:
        """
        Move the robot to target coordinates using grid movement.
        First calculates and turns to target direction, then moves straight.

        Args:
            target_x (int): Target X coordinate in millimeters
            target_y (int): Target Y coordinate in millimeters
        """
        # Get current position
        current_x, current_y, current_phi = self.actuator.position

        # Calculate direction to target
        dx = target_x - current_x
        dy = target_y - current_y

        # Calculate target angle in degrees
        target_degree = int(math.degrees(math.atan2(dy, dx)) % 360)

        # Turn to face target
        logger.info(f"{self.actuator.robot_name} {self.actuator.robot_id}: Turning to {target_degree:.1f}°")
        self.grid_turn(current_phi, target_degree)

        # Calculate distance to target
        distance = int(math.sqrt(dx * dx + dy * dy))
        logger.info(f"{self.actuator.robot_name} {self.actuator.robot_id}: Moving {distance} mm")

        # Move straight to target
        self.grid_straight(target_x, target_y)

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

        # Calculate target position based on direction
        if direction in ('west', 'southwest', 'south', 'southeast'):
            distance_x = -distance
        else:  # east, northeast, north, northwest
            distance_x = distance

        if direction in ('south', 'southeast', 'southwest'):
            distance_y = -distance
        else:  # north, northeast, northwest
            distance_y = distance

        # Adjust distance for diagonal movements
        if direction in ('northeast', 'northwest', 'southwest', 'southeast'):
            distance_x = int(distance_x * 0.707)  # cos(45°)
            distance_y = int(distance_y * 0.707)  # sin(45°)

        # Calculate final target position
        target_x = x + (distance_x if direction in ('east', 'west') else  # E, W
                        0 if direction in ('north', 'south') else  # N, S
                        distance_x)  # Diagonal

        target_y = y + (distance_y if direction in ('north', 'south') else  # N, S
                        0 if direction in ('east', 'west') else  # E, W
                        distance_y)  # Diagonal

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
        degree_to_turn = diff if diff <= 180 else diff - 360
        logger.info(f"Turning to target: {degree_to_turn} degrees remaining")
        while abs(degree_to_turn) > 5:
            if degree_to_turn > 0:
                VWTurn(5, 100)
            else:
                VWTurn(-5, 100)
            VWWait()
            _, _, _, _, phi = self._process_sensors()
            diff = ((target_degree - phi) % 360)
            degree_to_turn = diff if diff <= 180 else diff - 360
        VWWait()

    def grid_straight(self, target_x: int, target_y: int):
        """
        Move the robot straight in a grid pattern.

        Args:
            target_x (int): Target x coordinate in mm
            target_y (int): Target y coordinate in mm
        Returns:
            None
        """
        distance = calculate_distance(self.actuator.position.x, self.actuator.position.y, target_x, target_y)
        pre_distance = distance
        while distance <= pre_distance and is_movement_safe(self.actuator.scan):
            pre_distance = distance
            VWStraight(10, 100)
            VWWait()
            scan, img, x, y, phi = self._process_sensors()
            distance = calculate_distance(x, y, target_x, target_y)
        VWWait()

    def run_agent_with_action(self):
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
            response = None
            if not self.robot_state_record.action_queue:
                response = self._process_agent(response_format=RobotAction)
            action_data = response.get('response').get('action_queue', [])

            for action in action_data:
                self.robot_state_record.action_queue.append(
                    RobotAction(direction=action['direction'], distance=action['distance']))

            # move robot by popping next action in queue
            next_action = self.robot_state_record.action_queue.pop(0)
            next_action = RobotAction(direction=next_action.direction, distance=next_action.distance)
            self.move_grid(next_action.distance, next_action.direction)

    def run_agent(self):
        """
        Run the embodied agent with the given actuator.
        """
        # Search and rescue mission
        self._process_sensors()
        self.history_positions.append((self.actuator.position.x, self.actuator.position.y, self.actuator.position.phi))
        while self.step < MAXIMUM_STEP and self.target_remaining > 0:
            self.step += 1
            # update and check search mission status
            if self._check_search_mission_status():
                logger.info(f"Mission completed at step {self.step}")
                return

            # if action queue is empty, run process agent
            # TODO: needs more conditions to trigger process agent

            response = self._process_agent(response_format=WayPointList)
            waypoint_list = response.get('response').get('waypoint_list', [])
            logger.info(f"Waypoint list: {waypoint_list}")
            for waypoint in waypoint_list:
                self.move_to_target(waypoint.get('x'), waypoint.get('y'))
                self.history_positions.append((self.actuator.position.x, self.actuator.position.y, self.actuator.position.phi))
