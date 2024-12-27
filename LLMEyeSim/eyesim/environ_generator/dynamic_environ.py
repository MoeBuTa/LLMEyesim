from LLMEyeSim.eyesim.environ_generator import *


class DynamicEnviron(EnvironGenerator):
    def __init__(self):
        super().__init__(env_name="dynamic-environ")

    def generate_random_sim(self):
        indices = random.sample(range(len(self.dynamic_obstacles)), 2)
        content = f"""
# world "{self.world_file}"

settings VIS TRACE

# Robots
{self.dynamic_obstacles[indices[0]]} swarm.py

{self.dynamic_obstacles[indices[1]]} swarm.py

{random.choices(self.llm_robot)[0]} s4.py

# Objects
{random.choices(self.target)[0]}
        """
        with open(self.file_path, "w") as f:
            f.write(content)
