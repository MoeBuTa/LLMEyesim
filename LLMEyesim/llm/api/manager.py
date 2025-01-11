from typing import Any, Dict

from LLMEyesim.llm.api.base import BaseLLM
from LLMEyesim.llm.api.cloud_llm import CloudLLM
from LLMEyesim.llm.api.exceptions import InvalidLLMType
from LLMEyesim.llm.api.ollama_llm import OllamaLLM


class LLMManager:
    def __init__(self, llm_name: str, llm_type: str, **kwargs):
        """
        Initialize LLMManager with a specific llm.

        Args:
            llm_name: Name for the llm (e.g., 'gpt-4', 'gpt-4-turbo')
            llm_type: Type of llm ('cloud', 'quantization', or 'hf')
        """
        self.llm_types = {
            "cloud": CloudLLM,
            "ollama": OllamaLLM,
        }

        self.llm = self._init_llm(llm_name, llm_type, **kwargs)

    def _init_llm(self, llm_name: str, llm_type: str, **kwargs) -> BaseLLM:
        llm_type = llm_type.lower()
        if llm_type not in self.llm_types:
            raise InvalidLLMType(
                f"Invalid llm type. Must be one of: {', '.join(self.llm_types.keys())}"
            )

        llm_class = self.llm_types[llm_type]
        return llm_class(llm_name, llm_type, **kwargs)

    def process(self,**kwargs) -> Any:
        return self.llm.process(**kwargs)

    def process_v2(self, **kwargs) -> Any:
        return self.llm.process_v2(**kwargs)

    def get_llm_info(self) -> Dict[str, str]:
        return {
            "name": self.llm.name,
            "type": type(self.llm).__name__,
            "config": getattr(self.llm, "model", None),
        }
