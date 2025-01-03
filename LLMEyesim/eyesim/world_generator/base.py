import os

from loguru import logger

from LLMEyesim.utils.constants import EYESIM_DIR


class WorldGenerator:
    def __init__(self, world_name: str):
        logger.info(f"Initializing WorldGenerator with world name: {world_name}")
        self.world_name = world_name
        self.sim_file = f"{EYESIM_DIR}default.sim"
        self.world_file = f"{EYESIM_DIR}world.wld"
        self.labbot_script = f"{EYESIM_DIR}labbot.py"
        self.s4_script = f"{EYESIM_DIR}s4.py"
        self.llm_robot = ["S4 999 500 90", "S4 1009 1133 89"]
        self.target = ["Can 1716 1784 90", "Can 179 1765 90", "Can 273 225 90", "Can 1766 129 90"]
        self.dynamic_obstacles = ["LabBot 399 881 0", "LabBot 1441 1579 0", "LabBot 1200 253 0"]
        self.static_obstacles = ["Soccer 1362 600 90",
                                 "Soccer 509 442 90",
                                 "Soccer 1782 663 90",
                                 "Soccer 815 1742 90",
                                 "Soccer 1745 1115 90"]

        logger.debug("Initialized file paths and object positions")
        self._init_world()
        self._init_labbot_script()
        self._init_s4_script()
        logger.info("WorldGenerator initialization complete")

    def _init_s4_script(self):
        logger.info("Initializing S4 script")
        try:
            s4_script = f"""#!/usr/bin/env python3
from eye import *
from random import *

SAFE = 300
PSD_FRONT = 1
PSD_LEFT  = 2
PSD_RIGHT = 3

img = []
stop = False
while not stop:
        VWWait()

            """
            with open(f"{self.s4_script}", "w") as f:
                f.write(s4_script)
            os.chmod(self.s4_script, 0o755)
            logger.success(f"S4 script written successfully to {self.s4_script}")
        except Exception as e:
            logger.error(f"Failed to write S4 script: {str(e)}")
            raise

    def _init_labbot_script(self):
        logger.info("Initializing labbot script")
        try:
            labbot_script = f"""#!/usr/bin/env python3
from eye import *
from random import *

SAFE = 200
PSD_FRONT = 1
PSD_LEFT  = 2
PSD_RIGHT = 3

img = []
stop = False
id = OSMachineID() # unique ID
# NOTE: Only 1 robot (ID==1) will use LCD or KEYs
ME = (id==1)
# testing
LCDSetPrintf(20,0, "my id %d", id)
print( "my id %d\\n" % id)  # to console

if ME: LCDMenu("", "", "", "END")
CAMInit(QVGA)

while not stop:
    img = CAMGet()    # demo
    if ME: LCDImage(img)  # only
    f = PSDGet(PSD_FRONT)
    l = PSDGet(PSD_LEFT)
    r = PSDGet(PSD_RIGHT)
    if ME: LCDSetPrintf(18,0, "PSD L%3d F%3d R%3d", l, f, r)
    if l>SAFE and f>SAFE and r>SAFE:
        VWStraight( 100, 200) # 100mm at 10mm/s
    else:
        VWStraight(-25, 50)   # back up
        VWWait()
        dir = int(((random() - 0.5))*180)
        LCDSetPrintf(19,0, "Turn %d", dir)
        VWTurn(180, 45)      # turn random angle
        VWWait()
        if ME: LCDSetPrintf(19,0, "          ")
    OSWait(100)
    if ME: stop = (KEYRead() == KEY4)

                """
            with open(self.labbot_script, "w") as f:
                f.write(labbot_script)
            os.chmod(self.labbot_script, 0o755)
            logger.success(f"Labbot script written successfully to {self.labbot_script}")
        except Exception as e:
            logger.error(f"Failed to write labbot script: {str(e)}")
            raise

    def _init_world(self):
        logger.info("Initializing world file")
        try:
            wld_content = f"""
floor 2000 2000
0 2000 0 0
2000 2000 2000 0
2000 0 0 0
                """
            with open(self.world_file, "w") as f:
                f.write(wld_content)
            os.chmod(self.world_file, 0o755)
            logger.success(f"World file written successfully to {self.world_file}")
        except Exception as e:
            logger.error(f"Failed to write world file: {str(e)}")
            raise

    def generate_sim(self):
        logger.info("Attempting to generate simulation")
        raise NotImplementedError("generate_sim method not implemented yet")