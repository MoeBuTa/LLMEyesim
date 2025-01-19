from typing import List, Union

from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.eyesim.actuator.config import GRID_DIRECTION
from LLMEyesim.eyesim.actuator.models import Position
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.eyesim.utils.lidar_detection import (
    calculate_object_positions,
    detect_obstacles,
    update_object_positions,
)
from LLMEyesim.eyesim.utils.models import ObjectPosition, ObstacleRegion
from LLMEyesim.integration.config import MAXIMUM_STEP
from LLMEyesim.integration.models import (
    ExplorationRecord,
    ExplorationRecordList,
    LLMRecord,
    RobotAction,
    RobotStateRecord,
)
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent
from LLMEyesim.utils.constants import LOG_DIR
from datetime import datetime

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

        # obstacle regions by step
        self.obstacle_region_detected: List[ObstacleRegion] = []

        # exploration records by step
        self.exploration_records: ExplorationRecordList = ExplorationRecordList(records=[])
        self.step = 0

        # search and rescue mission
        self.reached_targets = []
        self.target_list = [item for item in world_items if item.item_type == "target"]
        self.target_remaining = len(self.target_list)

        # Safety Validation
        self.safety_flag = True
        self.reason = ""

        logger.success(
            f"Embodied agent initialized with {self.agent.llm_name}-{self.actuator.robot_name}-{self.actuator.robot_id}-{self.target_list}")

    def _update_records(self, **kwargs):
        if "new_llm_record" in kwargs:
            logger.info(f"Updating LLM record at step {self.step}")
            self.llm_records.append(kwargs["new_llm_record"])
        if "new_exploration_record" in kwargs:
            logger.info(f"Updating exploration record at step {self.step}")
            self.exploration_records.records.append(kwargs["new_exploration_record"])

    def _process_sensors(self, scan: List[int]):
        # process sensors and get environment information
        logger.info(f"Processing sensors at step {self.step}")

        # Object detection
        current_position = self.robot_state_record.positions[-1]
        new_object_detected = calculate_object_positions(robot_pos=(current_position.x, current_position.y),
                                                         objects=self.world_items, lidar_data=scan)
        self.object_detected = update_object_positions(new_object_detected, self.object_detected)
        logger.info(f"Detected objects: {self.object_detected}")

        # Obstacle region detection
        self.obstacle_region_detected = detect_obstacles(lidar_data=scan)
        logger.info(f"Detected obstacles: {self.obstacle_region_detected}")

        # Exploration record
        new_exploration_record = ExplorationRecord(object_positions=self.object_detected,
                                                   obstacle_regions=self.obstacle_region_detected,
                                                   reached_targets=self.reached_targets, step=self.step)
        self._update_records(new_exploration_record=new_exploration_record)

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

    @staticmethod
    def _is_movement_safe(
            lidar_data: List[int],
            direction: str,
            distance: int,
            angle_threshold: int = 15,  # Check ±15 degrees from movement direction
            safety_margin: int = 100  # Additional safety buffer (units)
    ) -> tuple[bool, str]:
        """
        Check if moving in a specific direction for a given distance is safe.

        Args:
            lidar_data: 360-degree lidar readings (integer values)
            direction: Direction of movement ('north', 'east', etc.)
            distance: Intended movement distance
            angle_threshold: Degrees to check on either side of movement direction
            safety_margin: Additional safety buffer distance

        Returns:
            tuple[bool, str]: (is_safe, reason)
            - is_safe: True if movement is safe, False otherwise
            - reason: Description of why movement is unsafe (if applicable)
        """
        if direction not in GRID_DIRECTION:
            return False, f"Invalid direction: {direction}"

        # Get center angle for the direction
        center_angle = GRID_DIRECTION[direction]

        # Calculate the range of angles to check
        start_angle = abs(center_angle - angle_threshold) % 360
        end_angle = (center_angle + angle_threshold) % 360

        # Get the minimum distance reading in the movement path
        if start_angle <= end_angle:
            scan_range = range(start_angle, end_angle + 1)
        else:
            # Handle wrap-around case (e.g., checking around 0/360 degrees)
            scan_range = list(range(start_angle, 360)) + list(range(0, end_angle + 1))

        # Check each angle in the range
        min_distance = float('inf')
        min_distance_angle = None

        for angle in scan_range:
            if lidar_data[angle] < min_distance:
                min_distance = lidar_data[angle]
                min_distance_angle = angle

        # Check if the path is clear
        safe_distance = distance + safety_margin
        logger.info(f"Checking movement safety: min_distance={min_distance}, safe_distance={safe_distance}, scan_range={scan_range}, direction={direction}")
        if min_distance <= safe_distance:
            return False, (
                f"Obstacle detected at angle {min_distance_angle}° "
                f"at distance {min_distance} units."
                f"Movement requires {safe_distance} units clearance "
                f"(requested distance {distance} + safety margin {safety_margin})"
            )

        return True, "Path is clear"

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
                if obj.item_id == target.item_id and self.actuator.calculate_distance(pos.x, pos.y, obj.x,
                                                                                      obj.y) < 100:
                    self.target_remaining -= 1
                    self.reached_targets.append(target.item_id)
        return False

    def run_agent(self):
        """
        Run the embodied agent with the given actuator.
        """
        # Search and rescue mission
        while self.step < MAXIMUM_STEP and self.target_remaining > 0:
            self.step += 1

            # get latest status and update records
            self.robot_state_record.positions.append(self.actuator.position)
            img, scan = self.actuator.img, self.actuator.scan
            self._process_sensors(scan=scan)

            # update and check search mission status
            if self._check_search_mission_status():
                logger.info(f"Mission completed at step {self.step}")
                return

            logger.info(
                f"Running {self.actuator.robot_name}-{self.actuator.robot_id} at {str(self.actuator.position)} with step {self.step}")

            # if action queue is empty, run process agent
            # TODO: needs more conditions to trigger process agent
            if not self.robot_state_record.action_queue or not self.safety_flag:
                self._process_agent()

            # move robot by popping next action in queue
            next_action = self.robot_state_record.action_queue.pop(0)
            logger.info(f"Next action: {next_action}")
            self.safety_flag, detail = self._is_movement_safe(scan, next_action.direction, next_action.distance)
            next_action = RobotAction(direction=next_action.direction, distance=next_action.distance,
                                      execution_status=self.safety_flag, detail=detail)
            if self.safety_flag:
                self.actuator.move_grid(next_action.distance, next_action.direction)
