from abc import abstractmethod
from typing import Any


class BaseLLM:
    def __init__(self, name: str, llm_type: str):
        self.name = name
        self.llm_type = llm_type


    @abstractmethod
    def process(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def process_v2(self, **kwargs) -> Any:
        pass