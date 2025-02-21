from loguru import logger

from LLMEyesim.eyesim.generator.base import WorldGenerator
from LLMEyesim.eyesim.generator.objects.crate import CRATE_LOCATIONS
from LLMEyesim.eyesim.generator.objects.robot import ROBOT_LOCATIONS
from LLMEyesim.eyesim.generator.objects.target import TARGET_LOCATIONS


class DemoWorld(WorldGenerator):
    def __init__(self, world_name: str, llm_name: str = "gpt-4o-mini"):
        super().__init__(world_name=world_name, llm_name=llm_name)
        # Spawn points in corners (with some offset from walls)
        self.spawn_points = ROBOT_LOCATIONS
        # Target positions (Red Cans) - placed at key locations
        self.target_positions = TARGET_LOCATIONS
        # Crate positions - placed as obstacles along paths to targets
        self.crate_locations = CRATE_LOCATIONS

    def init_sim(self, num_s4: int = 1, num_labbot: int = 0,
                 num_targets: int = 4):
        """
        Initialize the simulation with specified configuration.

        Args:
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

        # Add obstacles
        for x, y in self.crate_locations:
            self.create_object(object_name="Crate1", object_type='obstacle', x=x, y=y, angle=90)

        self.write_robot_script()
        self.generate_sim_file()

        logger.info(f"DemoWorld simulation generation complete")
        logger.info(f"Created {num_s4} S4 robots and {num_labbot} LabBot robots")
