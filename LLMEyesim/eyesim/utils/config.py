import numpy as np

from LLMEyesim.eyesim.utils.models import CardinalDirection

CARDINAL_DIRECTION_LUT = np.array([
    next(direction for angle, direction in {
        0: CardinalDirection.NORTH,
        45: CardinalDirection.NORTHEAST,
        90: CardinalDirection.EAST,
        135: CardinalDirection.SOUTHEAST,
        180: CardinalDirection.SOUTH,
        225: CardinalDirection.SOUTHWEST,
        270: CardinalDirection.WEST,
        315: CardinalDirection.NORTHWEST
    }.items() if abs((i - angle + 180) % 360 - 180) == min(abs((i - a + 180) % 360 - 180) for a in {0, 45, 90, 135, 180, 225, 270, 315}))
    for i in range(360)
])
