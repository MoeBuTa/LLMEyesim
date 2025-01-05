#!/usr/bin/env python3
from eye import *
from random import *

SAFE = 200
PSD_FRONT = 1
PSD_LEFT  = 2
PSD_RIGHT = 3

img = []
stop = False

while not stop:
    f = PSDGet(PSD_FRONT)
    l = PSDGet(PSD_LEFT)
    r = PSDGet(PSD_RIGHT)
    if l>SAFE and f>SAFE and r>SAFE:
        VWStraight( 100, 200) # 100mm at 10mm/s
    else:
        VWStraight(-25, 50)   # back up
        VWWait()
        dir = int(((random() - 0.5))*180)
        VWTurn(180, 45)      # turn random angle
        VWWait()
    OSWait(100)
                