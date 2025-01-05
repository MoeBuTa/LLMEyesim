from loguru import logger

from LLMEyesim.eyesim.generator.base import WorldGenerator



class DemoWorld(WorldGenerator):
    def __init__(self, world_name: str):
        super().__init__(world_name=world_name)

    def init_sim(self):
        labbot = self.create_robot(robot_name="LabBot", x=229, y=591, angle=20, script="labbot.py")
        s4 = self.create_robot(robot_name="S4", x=432, y=1659, angle=0, script="s4.py")
        robots = labbot + s4
        
        can = self.create_object(object_name="Can", x=1663, y=274, angle=90)
        soccer1 = self.create_object(object_name="Soccer", x=229, y=1391, angle=90)
        soccer2 = self.create_object(object_name="Soccer", x=1679, y=1525, angle=90)
        soccer3 = self.create_object(object_name="Soccer", x=1600, y=700, angle=0)
        objects = can + soccer1 + soccer2 + soccer3
        
        self.generate_sim(world_file=self.world_file, robots=robots, objects=objects)

        logger.info("DemoWorld simulation generation complete")
