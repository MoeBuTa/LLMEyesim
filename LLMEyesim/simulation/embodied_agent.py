from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent


class EmbodiedAgent:
    def __init__(self, agent: ExecutiveAgent, actuator: RobotActuator, **kwargs):
        self.agent = agent
        self.actuator = actuator



    def run_agent(self):
        logger.info(f"Running embodied agent: {self.actuator.robot_name}")
