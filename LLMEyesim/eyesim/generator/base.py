import os
from pathlib import Path
from typing import List, Literal

from loguru import logger

from LLMEyesim.eyesim.generator.config import (
    AVAILABLE_OBJECTS,
    AVAILABLE_ROBOTS,
    RANDOM_CAN_LOCATIONS,
    RANDOM_LABBOT_LOCATIONS,
    RANDOM_S4_LOCATIONS,
    RANDOM_SOCCER_LOCATIONS,
)
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.utils.constants import EYESIM_DIR, SCRIPT_DIR, WORLD_DIR


class WorldGenerator:
    def __init__(self, world_name: str, llm_name: str = "gpt-4o-mini"):
        logger.info(f"Initializing WorldGenerator with world name: {world_name}")
        self.sim_file = f"{EYESIM_DIR}/default.sim"
        self.world_file = f"{WORLD_DIR}/{world_name}.wld"

        self.llm_name = llm_name

        self.items: List[WorldItem] = []
        self.robots: List[WorldItem] = []
        self.robot_settings: str = ""
        self.objects: List[WorldItem] = []
        self.object_settings: str = ""
        self.world_name = world_name
        self._init_execute_permission()

        # Legacy code
        self.llm_robot = RANDOM_S4_LOCATIONS
        self.target = RANDOM_CAN_LOCATIONS
        self.dynamic_obstacles = RANDOM_LABBOT_LOCATIONS
        self.static_obstacles = RANDOM_SOCCER_LOCATIONS

    @staticmethod
    def _init_execute_permission():
        """Make all files in the script and world directories executable with 0o777 permissions."""
        dirs = [Path(SCRIPT_DIR), Path(WORLD_DIR)]

        for dir_path in dirs:
            try:
                for file in dir_path.iterdir():
                    if file.is_file():
                        file.chmod(0o777)
                        print(f"Made {file.name} executable")
            except PermissionError:
                print(f"Permission denied for files in {dir_path}")
            except Exception as e:
                print(f"Error processing {dir_path}: {e}")

    def init_sim(self, **kwargs):
        raise NotImplementedError

    def create_robot(self, robot_name: str, x: int, y: int, angle: int) -> None:
        if robot_name not in AVAILABLE_ROBOTS:
            raise ValueError(f"Robot {robot_name} is not in the list of available robots")
        item_id = len(self.robots) + 1
        self.robots.append(WorldItem(item_id=item_id, item_name=robot_name, item_type='robot', x=x, y=y, angle=angle))
        self.robot_settings += f"{robot_name} {x} {y} {angle} {SCRIPT_DIR}/{self.llm_name}_{robot_name}_{item_id}.py\n"

    def create_object(self, object_name: str, object_type: Literal['target', 'obstacle', 'robot'], x: int, y: int,
                      angle: int) -> None:
        if object_name not in AVAILABLE_OBJECTS:
            raise ValueError(f"Object {object_name} is not in the list of available objects")
        item_id = len(self.robots) + len(self.objects) + 1
        self.objects.append(WorldItem(item_id=item_id, item_name=object_name, item_type=object_type, x=x, y=y, angle=angle))
        self.object_settings += f"{object_name} {x} {y} {angle}\n"

    def write_robot_script(self) -> None:
        self.items = self.robots + self.objects
        logger.debug("Creating robot script content")
        try:
            for i, robot in enumerate(self.robots):
                content = f"""#!/Users/wenxiao/miniconda3/envs/llmeyesim/bin/python

from LLMEyesim.eyesim.actuator.actuator import RobotActuator
from LLMEyesim.eyesim.generator.models import WorldItem
from LLMEyesim.llm.agents.executive_agent import ExecutiveAgent
from LLMEyesim.integration.embodied_agent import EmbodiedAgent

if __name__ == '__main__':
    
    world_items = {self.items}
    agent = ExecutiveAgent(llm_name='{self.llm_name}', llm_type="cloud")
    actuator = RobotActuator(robot_id={i + 1}, robot_name='{robot.item_name}')
    embodied_agent = EmbodiedAgent(agent, actuator, world_items)
    embodied_agent.run_agent()
"""
                script_file = f"{SCRIPT_DIR}/{self.llm_name}_{robot.item_name}_{i + 1}.py"
                with open(script_file, "w") as f:
                    f.write(content)
                os.chmod(script_file, 0o777)
                logger.success(f"Successfully wrote and made executable script file to {script_file}")
        except Exception as e:
            logger.error(f"Failed to write script file: {str(e)}")
            raise

    def generate_sim_file(self):
        logger.debug("Creating simulation file content")
        try:
            content = f"""
# world 
world {self.world_file}

settings VIS TRACE

# Robots
{self.robot_settings}

# Objects
{self.object_settings}
            """
            sim_file = f"{EYESIM_DIR}/default.sim"
            with open(sim_file, "w") as f:
                f.write(content)
            os.chmod(sim_file, 0o777)
            logger.success(f"Successfully wrote and made executable simulation file to {sim_file}")
        except Exception as e:
            logger.error(f"Failed to write simulation file: {str(e)}")
            raise
