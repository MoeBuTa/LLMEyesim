from LLMEyeSim.eyesim.environ_generator import *


class FreeEnviron(EnvironGenerator):
    def __init__(self):
        super().__init__(env_name="free-environ")

    def generate_random_sim(self):
        content = f"""
# world "{self.world_file}"
settings VIS TRACE

# Robots
{random.choices(self.llm_robot)[0]}

# Objects
{random.choices(self.target)[0]}
        """
        with open(self.file_path, "w") as f:
            f.write(content)
