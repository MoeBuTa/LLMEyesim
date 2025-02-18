from loguru import logger
from enum import Enum
from typing import Tuple

from LLMEyesim.eyesim.generator.base import WorldGenerator
from LLMEyesim.eyesim.worlds.crate_locations import CRATE_LOCATIONS


class WorldType(Enum):
    EMPTY = 1  # No obstacles
    WALLS = 2  # Only walls
    COMPLEX = 3  # Walls and soccer balls


class DemoWorld(WorldGenerator):
    def __init__(self, world_name: str, llm_name: str = "gpt-4o-mini"):
        super().__init__(world_name=world_name, llm_name=llm_name)
        # Spawn points in corners (with some offset from walls)
        self.spawn_points = [
            (300, 300),  # Bottom-left corner
            (5700, 300),  # Bottom-right corner
            (300, 5700),  # Top-left corner
            (5700, 5700)  # Top-right corner
        ]

        # Target positions (Red Cans) - placed at key locations
        self.target_positions = [
            (3000, 5500),  # Top middle
            (500, 3000),  # Left middle
            (5500, 3000),  # Right middle
            (3000, 500)  # Bottom middle
        ]

        # Soccer positions - placed as obstacles along paths to targets
        self.crate_locations = CRATE_LOCATIONS

    def init_sim(self, world_type: WorldType=WorldType.COMPLEX, num_s4: int = 3, num_labbot: int = 1,
                 num_targets: int = 4):
        """
        Initialize the simulation with specified configuration.

        Args:
            world_type: Type of world (EMPTY, WALLS, or COMPLEX)
            num_s4: Number of S4 robots (0-4)
            num_labbot: Number of LabBot robots (0-4)
            num_targets: Number of targets/Cans (1-4)
        """
        total_robots = num_s4 + num_labbot
        if total_robots < 1 or total_robots > 4:
            raise ValueError("Total number of robots (S4 + LabBot) must be between 1 and 4")
        if not 1 <= num_targets <= 4:
            raise ValueError("Number of targets must be between 1 and 4")

        spawn_idx = 0  # Keep track of spawn point index

        # Create S4 robots
        for i in range(num_s4):
            if spawn_idx >= len(self.spawn_points):
                break
            x, y = self.spawn_points[spawn_idx]
            self.create_robot(robot_name="S4", x=x, y=y, angle=0)
            spawn_idx += 1

        # Create LabBot robots
        for i in range(num_labbot):
            if spawn_idx >= len(self.spawn_points):
                break
            x, y = self.spawn_points[spawn_idx]
            self.create_robot(robot_name="LabBot", x=x, y=y, angle=0)
            spawn_idx += 1

        # Create targets (Cans)
        for i in range(num_targets):
            x, y = self.target_positions[i]
            self.create_object(object_name="Can", object_type='target', x=x, y=y, angle=90)

        # Add obstacles based on world type
        if world_type == WorldType.COMPLEX:
            for x, y in self.crate_locations:
                self.create_object(object_name="Crate1", object_type='obstacle', x=x, y=y, angle=90)

        self.write_robot_script()
        self.generate_sim_file()

        logger.info(f"DemoWorld simulation generation complete - World Type: {world_type.name}")
        logger.info(f"Created {num_s4} S4 robots and {num_labbot} LabBot robots")
