from LLMEyeSim.eyesim.environ_generator import *


class StaticEnviron(EnvironGenerator):
    def __init__(self):
        super().__init__(env_name="static-environ")

    def generate_random_sim(self):
        indices = random.sample(range(len(self.static_obstacles)), 4)
        content = f"""
# world "{self.world_file}"
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
        with open(self.file_path, "w") as f:
            f.write(content)

    def generate_sim(self):
        swarm_script = f"""
#!/usr/bin/env python3
from eye import *
from random import *

SAFE = 200
PSD_FRONT = 1
PSD_LEFT  = 2
PSD_RIGHT = 3

def main():
    img = []
    stop = False
    id = OSMachineID() # unique ID
    # NOTE: Only 1 robot (ID==1) will use LCD or KEYs
    ME = (id==1)
    # testing
    LCDSetPrintf(20,0, "my id %d", id)
    print( "my id %d\n" % id)  # to console

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

main()
        """
        with open(self.script_file, "w") as f:
            f.write(swarm_script)

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

        content = f"""
# world {self.world_file}
settings TRACE

# Robots
LabBot 229 591 20 swarm.py
S4 432 1659 0 s4.py

# Objects
Can 1663 274 90
Soccer 229 1391 90
Soccer 1679 1525 90
Soccer 1600 700 0

                """
        with open(self.file_path, "w") as f:
            f.write(content)
