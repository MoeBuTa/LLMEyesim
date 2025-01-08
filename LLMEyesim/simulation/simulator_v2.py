from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent
from LLMEyesim.simulation.embodied_agent import EmbodiedAgent
from LLMEyesim.simulation.models import SimulatorV2Config
import multiprocessing


class SimulatorV2:
    def __init__(self, **kwargs):
        self.config = SimulatorV2Config(**kwargs)
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize simulator components with error handling"""
        try:
            self.mission_name = self.config.mission_name
            self.world_items = self.config.world_items
        except Exception as e:
            logger.error(f"Failed to initialize simulator components: {str(e)}")
            raise RuntimeError(f"Simulator initialization failed: {str(e)}")

    @staticmethod
    def run_embodied_agent(llm_name: str, llm_type: str, i: int, item_name: str):
        """Run the embodied agent"""
        agent = ExecutiveAgent(llm_name=llm_name, llm_type=llm_type)
        actuator = RobotActuator(robot_id=i, robot_name=item_name)
        EmbodiedAgent(agent, actuator).run_agent()

    def run(self):
        """Run the simulator"""
        processes = []
        for i, item in enumerate(self.world_items):
            if item.item_type == "robot":
                p = multiprocessing.Process(target=self.run_embodied_agent, args=(self.config.llm_name, self.config.llm_type, i, item.item_name))
                processes.append(p)
                p.start()

        for p in processes:
            p.join()
