from typing import List

from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.eyesim.actuator.models import Position
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.eyesim.utils.lidar_detection import calculate_object_positions, detect_obstacles
from LLMEyesim.integration.config import MAXIMUM_STEP
from LLMEyesim.integration.models import ExplorationRecord, LLMRecord, RobotStateRecord, RobotAction
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent
from LLMEyesim.utils.constants import LOG_DIR

logger.add(f"{LOG_DIR}/running_logs.txt", rotation="10 MB", format="{time} | {level} | {message}")


class EmbodiedAgent:
    def __init__(self, agent: ExecutiveAgent, actuator: RobotActuator, world_items: List[WorldItem], **kwargs):
        self.world_items = world_items
        self.agent = agent
        self.actuator = actuator
        self.action_queue: List[RobotAction] = []
        self.llm_records: List[LLMRecord] = []
        self.robot_records: List[RobotStateRecord] = []
        self.exploration_records: List[ExplorationRecord] = []
        self.step = 0

    def _update_records(self, **kwargs):
        if "new_llm_record" in kwargs:
            self.llm_records.append(kwargs["new_llm_record"])
        if "new_robot_state_record" in kwargs:
            self.robot_records.append(kwargs["new_robot_state_record"])
        if "new_exploration_record" in kwargs:
            self.exploration_records.append(kwargs["new_exploration_record"])

    def _process_sensors(self, current_position: Position, scan: List[int]):
        # process sensors and get environment information
        # TODO: Implement object detection and obstacle detection
        object_detected = calculate_object_positions((current_position.x, current_position.y), self.world_items, scan)
        obstacle_region_detected = detect_obstacles(scan)
        new_exploration_record = ""
        self._update_records(new_exploration_record=new_exploration_record)

        new_robot_state_record = RobotStateRecord(x=current_position.x, y=current_position.y, phi=current_position.phi,
                                                  action_queue=self.action_queue, step=self.step)
        self._update_records(new_robot_state_record=new_robot_state_record)

    def _process_agent(self):
        response = self.agent.process_v2(exploration_records=str([str(record) for record in self.exploration_records]),
                                         robot_state=str(self.robot_records[-1]))
        self.action_queue = response.get("action_queue", [])
        new_llm_record = LLMRecord(messages=response.messages, response=response.response, step=self.step)
        self._update_records(new_llm_record=new_llm_record)

    def run_agent(self, objects: List[WorldItem]):
        scan = self.actuator.scan
        if self.actuator.robot_name == "LabBot":
            logger.info([str(x) for x in scan])
        self.actuator.move_grid(0, direction="south")
        if self.actuator.robot_name == "LabBot":
            logger.info([str(x) for x in scan])

        while self.step < MAXIMUM_STEP:
            self.step += 1

            # get latest status and update records
            current_position = self.actuator.position
            img, scan = self.actuator.img, self.actuator.scan
            self._process_sensors(current_position=current_position, scan=scan)
            logger.info(
                f"Running {self.actuator.robot_name} - {self.actuator.robot_id} at {str(current_position)} with step {self.step}")

            # if action queue is empty, run process agent
            if not self.action_queue:
                self._process_agent()

            # move robot by popping next action in queue
            next_action = self.action_queue.pop(0)
            self.actuator.move_grid(next_action.distance, next_action.direction)
