import random

from LLMEyesim.eyesim.world_generator.base import WorldGenerator


class StaticWorld(WorldGenerator):
    def __init__(self, world_name: str):
        super().__init__(world_name=world_name)

    def generate_sim(self):
        indices = random.sample(range(len(self.static_obstacles)), 4)
        content = f"""
# world
world world.wld

settings VIS TRACE
# Robots
{random.choices(self.llm_robot)[0]}

# Objects
{random.choices(self.target)[0]}
{self.static_obstacles[indices[0]]}
{self.static_obstacles[indices[1]]}
{self.static_obstacles[indices[2]]}
{self.static_obstacles[indices[3]]}
        """
        with open(self.sim_file, "w") as f:
            f.write(content)