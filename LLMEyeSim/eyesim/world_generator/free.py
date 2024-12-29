import random

from LLMEyeSim.eyesim.world_generator.base import WorldGenerator


class FreeWorld(WorldGenerator):
    def __init__(self, world_name: str):
        super().__init__(world_name=world_name)

    def generate_sim(self):
        content = f"""
# world
world world.wld

settings VIS TRACE

# Robots
{random.choices(self.llm_robot)[0]}

# Objects
{random.choices(self.target)[0]}
        """
        with open(self.sim_file, "w") as f:
            f.write(content)
