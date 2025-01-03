from abc import abstractmethod
from typing import Any


class BaseAgent:
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type


    @abstractmethod
    def process(self, **kwargs) -> Any:
        pass
