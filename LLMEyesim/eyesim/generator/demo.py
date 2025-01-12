from loguru import logger

from LLMEyesim.eyesim.generator.base import WorldGenerator


class DemoWorld(WorldGenerator):
    def __init__(self, world_name: str, llm_name: str = "gpt-4o-mini"):
        super().__init__(world_name=world_name, llm_name=llm_name)

    def init_sim(self, **kwargs):
        # self.create_robot(robot_name="LabBot", x=229, y=591, angle=0)
        self.create_robot(robot_name="S4", x=432, y=1659, angle=0)

        self.create_object(object_name="Can", object_type='target', x=1663, y=274, angle=90)
        self.create_object(object_name="Soccer", object_type='obstacle', x=229, y=1391, angle=90)
        self.create_object(object_name="Soccer", object_type='obstacle', x=1679, y=1525, angle=90)
        self.create_object(object_name="Soccer", object_type='obstacle', x=1600, y=700, angle=0)

        self.write_robot_script()
        self.generate_sim_file()

        logger.info("DemoWorld simulation generation complete")
