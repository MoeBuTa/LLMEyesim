from dataclasses import dataclass


@dataclass(frozen=True)
class WorldItem:
    item_name: str
    item_type: str
    x: int = 0
    y: int = 0
    angle: int = 0
