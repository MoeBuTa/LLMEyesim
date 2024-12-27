from LLMEyeSim.eyesim.environ_generator import *


class StaticDynamicEnviron(EnvironGenerator):
    def __init__(self):
        super().__init__(env_name="static-dynamic-environ")

    def generate_random_sim(self):
        indices = random.sample(range(len(self.static_obstacles)), 3)
        content = f"""
# world "{self.world_file}"
settings VIS TRACE

# Robots


{random.sample(self.dynamic_obstacles, 1)[0]} swarm.py

{random.choices(self.llm_robot)[0]} s4.py

# Objects
{random.choices(self.target)[0]}
{self.static_obstacles[indices[0]]}
{self.static_obstacles[indices[1]]}
{self.static_obstacles[indices[2]]}
        """
        with open(self.file_path, "w") as f:
            f.write(content)
