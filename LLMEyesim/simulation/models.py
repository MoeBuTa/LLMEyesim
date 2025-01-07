from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for the simulator with immutable attributes"""
    task_name: str
    world_items: List = None
    llm_name: str = "gpt-4o-mini"
    llm_type: str = "cloud"
    attack: str = ""
    attack_rate: float = 0.5
    enable_defence: bool = False
    max_steps: int = 20
    red_detection_threshold: int = 100
    failure_retry_threshold: int = 3


@dataclass(frozen=True)
class SimulatorV2Config:
    """Configuration for the simulator with immutable attributes"""
    task_name: str
    world_items: List = None
    llm_name: str = "gpt-4o-mini"
    llm_type: str = "cloud"
