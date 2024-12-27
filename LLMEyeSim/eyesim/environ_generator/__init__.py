import random


class EnvironGenerator:
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.file_path = f"/opt/eyesim/eyesimX/{self.env_name}.sim"
        self.world_file = "box.maz"
        self.script_file = "/opt/eyesim/eyesimX/swarm.py"
        self.llm_robot = ["S4 999 500 90", "S4 1009 1133 89"]
        self.target = ["Can 1716 1784 90", "Can 179 1765 90", "Can 273 225 90", "Can 1766 129 90"]
        self.dynamic_obstacles = ["LabBot 399 881 0", "LabBot 1441 1579 0", "LabBot 1200 253 0"]
        self.static_obstacles = ["Soccer 1362 600 90",
                                 "Soccer 509 442 90",
                                 "Soccer 1782 663 90",
                                 "Soccer 815 1742 90",
                                 "Soccer 1745 1115 90"
                                 ]

    def generate_random_sim(self):
        raise NotImplementedError
