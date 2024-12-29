import random

from LLMEyeSim.eyesim.world_generator.base import WorldGenerator


class DynamicWorld(WorldGenerator):
    def __init__(self, world_name: str):
        super().__init__(world_name=world_name)

    def generate_sim(self):
        indices = random.sample(range(len(self.dynamic_obstacles)), 2)
        content = f"""
# world
world world.wld

settings VIS TRACE

# Robots
{self.dynamic_obstacles[indices[0]]} swarm.py

{self.dynamic_obstacles[indices[1]]} swarm.py

{random.choices(self.llm_robot)[0]} s4.py

# Objects
{random.choices(self.target)[0]}
        """
        with open(self.sim_file, "w") as f:
            f.write(content)
