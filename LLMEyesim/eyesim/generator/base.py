from dataclasses import dataclass
import os
from pathlib import Path

from loguru import logger

from LLMEyesim.utils.constants import EYESIM_DIR, SCRIPT_DIR, WORLD_DIR


@dataclass(frozen=True)
class WorldItem:
    item_name: str
    item_type: str
    x: int = 0
    y: int = 0
    angle: int = 0


class WorldGenerator:
    def __init__(self, world_name: str):
        logger.info(f"Initializing WorldGenerator with world name: {world_name}")
        self.items: list[WorldItem] = []
        self.world_name = world_name
        self.sim_file = f"{EYESIM_DIR}/default.sim"
        self.world_file = f"{WORLD_DIR}/{world_name}.wld"
        self.robot_list = ['LabBot', 'S4']
        self.object_list = ['Can', 'Soccer']
        self.llm_robot = ["S4 999 500 90", "S4 1009 1133 89"]
        self.target = ["Can 1716 1784 90", "Can 179 1765 90", "Can 273 225 90", "Can 1766 129 90"]
        self.dynamic_obstacles = ["LabBot 399 881 0", "LabBot 1441 1579 0", "LabBot 1200 253 0"]
        self.static_obstacles = ["Soccer 1362 600 90",
                                 "Soccer 509 442 90",
                                 "Soccer 1782 663 90",
                                 "Soccer 815 1742 90",
                                 "Soccer 1745 1115 90"]

        logger.debug("Initialized file paths and object positions")
        self._init_execute_permission(SCRIPT_DIR)
        self._init_execute_permission(WORLD_DIR)
        logger.info("WorldGenerator initialization complete")

    def _init_execute_permission(self, path: str):
        """Make all files in the script directory executable with 0o777 permissions.

        Args:
            path: Path to the directory
        """
        dir_path = Path(path)
        for file in dir_path.iterdir():
            if file.is_file():
                os.chmod(str(file), 0o777)
                print(f"Made {file.name} executable")
        os.chmod(self.world_file, 0o777)

    def init_sim(self):
        raise NotImplementedError

    def create_robot(self, robot_name: str, x: int, y: int, angle: int, script: str) -> str:
        if robot_name not in self.robot_list:
            raise ValueError(f"Robot {robot_name} is not in the list of available robots")
        self.items.append(WorldItem(item_name=robot_name, item_type='robot', x=x, y=y, angle=angle))
        return f"{robot_name} {x} {y} {angle} {SCRIPT_DIR}/{script}\n"

    def create_object(self, object_name: str, x: int, y: int, angle: int) -> str:
        if object_name not in self.object_list:
            raise ValueError(f"Object {object_name} is not in the list of available objects")
        self.items.append(WorldItem(item_name=object_name, item_type='object', x=x, y=y, angle=angle))
        return f"{object_name} {x} {y} {angle}\n"

    @staticmethod
    def generate_sim(world_file: str, robots: str, objects: str):
        logger.debug("Creating simulation file content")
        try:
            content = f"""
# world 
world {world_file}

settings TRACE

# Robots
{robots}

# Objects
{objects}
            """
            sim_file = f"{EYESIM_DIR}/default.sim"
            with open(sim_file, "w") as f:
                f.write(content)
            os.chmod(sim_file, 0o777)
            logger.success(f"Successfully wrote and made executable simulation file to {sim_file}")
        except Exception as e:
            logger.error(f"Failed to write simulation file: {str(e)}")
            raise
