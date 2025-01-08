from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent
from LLMEyesim.utils.constants import LOG_DIR

logger.add(f"{LOG_DIR}/running_logs.txt", rotation="10 MB", format="{time} | {level} | {message}")

class EmbodiedAgent:
    def __init__(self, agent: ExecutiveAgent, actuator: RobotActuator, **kwargs):
        self.agent = agent
        self.actuator = actuator
        self.memory = []

    def run_agent(self):
        logger.info(f"Running embodied agent: {self.actuator.robot_name}")
        logger.info(f"Robot Position: {self.actuator.position}")


        self.actuator.move_grid(direction="south", distance=100)
        self.actuator.move_grid(direction="east", distance=100)
