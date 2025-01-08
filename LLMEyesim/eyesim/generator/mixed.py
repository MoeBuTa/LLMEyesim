import random

from LLMEyesim.eyesim.generator.base import WorldGenerator
from LLMEyesim.utils.constants import SCRIPT_DIR


class MixedWorld(WorldGenerator):
    def __init__(self, world_name: str):
        super().__init__(world_name=world_name)

    def init_sim(self, **kwargs):
        indices = random.sample(range(len(self.static_obstacles)), 3)
        content = f"""
world {self.world_file}

settings VIS TRACE

# Robots


{random.sample(self.dynamic_obstacles, 1)[0]} {SCRIPT_DIR}/labbot.py

{random.choices(self.llm_robot)[0]} {SCRIPT_DIR}/s4.py

# Objects
{random.choices(self.target)[0]}
{self.static_obstacles[indices[0]]}
{self.static_obstacles[indices[1]]}
{self.static_obstacles[indices[2]]}
        """
        with open(self.sim_file, "w") as f:
            f.write(content)
