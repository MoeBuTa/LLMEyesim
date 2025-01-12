from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WorldItem:
    item_id: int
    item_name: str
    item_type: Literal['robot', 'obstacle', 'target']
    x: int = 0
    y: int = 0
    angle: int = 0
