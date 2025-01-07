from loguru import logger

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.simulation.models import SimulatorV2Config


class SimulatorV2:
    def __init__(self, **kwargs):
        self.config = SimulatorV2Config(**kwargs)
        self._initialize_components()


    def _initialize_components(self) -> None:
        """Initialize simulator components with error handling"""
        try:
            self.items = self.config.items
            robot_id = next((i for i, item in enumerate(self.items) if item.item_name == "S4"), -1) + 1

            self.actuator = RobotActuator(robot_id, "S4")
            self.agent = ActionAgent(
                task_name=self.config.task_name,
                agent_name=self.config.agent_name,
                agent_type=self.config.agent_type
            )

        except Exception as e:
            logger.error(f"Failed to initialize simulator components: {str(e)}")
            raise RuntimeError(f"Simulator initialization failed: {str(e)}")



    def run(self):
        pass

