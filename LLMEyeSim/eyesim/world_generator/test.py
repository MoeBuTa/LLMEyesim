import os

from loguru import logger

from LLMEyeSim.eyesim.world_generator.base import WorldGenerator


class TestWorld(WorldGenerator):
    def __init__(self, world_name: str):
        logger.info(f"Initializing TestWorld with world name: {world_name}")
        super().__init__(world_name=world_name)
        logger.debug("TestWorld initialization complete")

    def generate_sim(self):
        logger.info("Generating simulation files for TestWorld")

        # Generate world file
        logger.debug("Creating world file content")
        try:
            wld_content = f"""
floor 2000 2000
0 2000 0 0
2000 2000 2000 0
2000 0 0 0
0 2000 2000 2000
1400 2000 1400 900
0 320 1100 320
            """

            with open(self.world_file, "w") as f:
                f.write(wld_content)
            # Make the file executable
            os.chmod(self.world_file, 0o777)  # rwxr-xr-x permissions
            logger.success(f"Successfully wrote and made executable world file to {self.world_file}")
        except Exception as e:
            logger.error(f"Failed to write world file: {str(e)}")
            raise

        # Generate simulation file
        logger.debug("Creating simulation file content")
        try:
            content = f"""
# world 
world world.wld

settings TRACE

# Robots
LabBot 229 591 20 labbot.py
S4 432 1659 0 s4.py


# Objects
Can 1663 274 90
Soccer 229 1391 90
Soccer 1679 1525 90
Soccer 1600 700 0
            """

            with open(self.sim_file, "w") as f:
                f.write(content)
            # Make the file executable
            os.chmod(self.sim_file, 0o777)  # rwxr-xr-x permissions
            logger.success(f"Successfully wrote and made executable simulation file to {self.sim_file}")
        except Exception as e:
            logger.error(f"Failed to write simulation file: {str(e)}")
            raise

        logger.info("TestWorld simulation generation complete")