from typing import List, Union

from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
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

        self.action_queue: List[RobotAction] = []

        self.llm_records: List[LLMRecord] = []

        self.robot_records: List[RobotStateRecord] = []

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

        logger.success(f"Embodied agent initialized with {self.agent.llm_name}-{self.actuator.robot_name}-{self.actuator.robot_id}-{self.target_list}")

    def _update_records(self, **kwargs):
        if "new_llm_record" in kwargs:
            logger.info(f"Updating LLM record at step {self.step}")
            self.llm_records.append(kwargs["new_llm_record"])
        if "new_robot_state_record" in kwargs:
            logger.info(f"Updating robot state record at step {self.step}")
            self.robot_records.append(kwargs["new_robot_state_record"])
        if "new_exploration_record" in kwargs:
            logger.info(f"Updating exploration record at step {self.step}")
            self.exploration_records.records.append(kwargs["new_exploration_record"])

    def _process_sensors(self, current_position: Position, scan: List[int]):
        # process sensors and get environment information
        logger.info(f"Processing sensors at step {self.step}")

        # Object detection
        new_object_detected = calculate_object_positions(robot_pos=(current_position.x, current_position.y),
                                                         objects=self.world_items, lidar_data=scan)
        self.object_detected = update_object_positions(new_object_detected, self.object_detected)
        logger.info(f"Detected objects: {self.object_detected}")

        # Obstacle region detection
        self.obstacle_region_detected = detect_obstacles(lidar_data=scan)
        logger.info(f"Detected obstacles: {self.obstacle_region_detected}")

        # Exploration record
        new_exploration_record = ExplorationRecord(object_positions=self.object_detected,
                                                   obstacle_regions=self.obstacle_region_detected, reached_targets=self.reached_targets, step=self.step)
        self._update_records(new_exploration_record=new_exploration_record)

        # Robot state record
        new_robot_state_record = RobotStateRecord(x=current_position.x, y=current_position.y, phi=current_position.phi,
                                                  action_queue=self.action_queue, step=self.step)
        self._update_records(new_robot_state_record=new_robot_state_record)

    def _process_agent(self):
        logger.info(f"Processing agent at step {self.step}")
        response = self.agent.process_v2(exploration_records=str(self.exploration_records),
                                         robot_state=str(self.robot_records[-1]))
        action_data = response.get('response').get('action_queue', [])
        self.action_queue = [
            RobotAction(direction=action['direction'], distance=action['distance'])
            for action in action_data
        ]
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
            current_position = self.actuator.position
            img, scan = self.actuator.img, self.actuator.scan
            self._process_sensors(current_position=current_position, scan=scan)

            # update and check search mission status
            if self._check_search_mission_status():
                logger.info(f"Mission completed at step {self.step}")
                return

            logger.info(
                f"Running {self.actuator.robot_name}-{self.actuator.robot_id} at {str(current_position)} with step {self.step}")

            # if action queue is empty, run process agent
            # TODO: needs more conditions to trigger process agent
            if not self.action_queue:
                self._process_agent()

            # move robot by popping next action in queue
            next_action = self.action_queue.pop(0)
            self.actuator.move_grid(next_action.distance, next_action.direction)
