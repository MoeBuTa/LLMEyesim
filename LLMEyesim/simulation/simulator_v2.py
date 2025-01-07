from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent
from LLMEyesim.simulation.models import SimulatorV2Config


class SimulatorV2:
    def __init__(self, **kwargs):
        self.config = SimulatorV2Config(**kwargs)
        self._initialize_components()


    def _initialize_components(self) -> None:
        """Initialize simulator components with error handling"""
        try:
            self.world_items = self.config.world_items
            robot_id = next((i for i, item in enumerate(self.world_items) if item.item_name == "S4"), -1) + 1
            self.actuator = RobotActuator(robot_id, "S4")
            self.agent = ExecutiveAgent(
                task_name=self.config.task_name,
                llm_name=self.config.llm_name,
                llm_type=self.config.llm_type
            )
        except Exception as e:
            logger.error(f"Failed to initialize simulator components: {str(e)}")
            raise RuntimeError(f"Simulator initialization failed: {str(e)}")



    def run(self):
        """Run the simulator"""
        pass

