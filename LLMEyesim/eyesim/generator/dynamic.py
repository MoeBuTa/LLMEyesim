import random

from LLMEyesim.eyesim.generator.base import WorldGenerator
from LLMEyesim.utils.constants import SCRIPT_DIR


class DynamicWorld(WorldGenerator):
    def __init__(self, world_name: str):
        super().__init__(world_name=world_name)

    def init_sim(self):
        indices = random.sample(range(len(self.dynamic_obstacles)), 2)

        content = f"""
# world
world {self.world_file}

settings VIS TRACE

# Robots
{self.dynamic_obstacles[indices[0]]} {SCRIPT_DIR}/labbot.py

{self.dynamic_obstacles[indices[1]]} {SCRIPT_DIR}/labbot.py

{random.choices(self.llm_robot)[0]} {SCRIPT_DIR}/s4.py

# Objects
{random.choices(self.target)[0]}
        """
        with open(self.sim_file, "w") as f:
            f.write(content)
