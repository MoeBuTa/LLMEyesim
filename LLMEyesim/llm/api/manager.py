from typing import Any, Dict

from LLMEyesim.llm.api.base import BaseAgent
from LLMEyesim.llm.api.cloud_agent import CloudAgent
from LLMEyesim.llm.api.exceptions import InvalidAgentType
from LLMEyesim.llm.api.ollama_agent import OllamaAgent


class AgentManager:
    def __init__(self, agent_name: str, agent_type: str, **kwargs):
        """
        Initialize AgentManager with a specific agent.

        Args:
            agent_name: Name for the agent (e.g., 'gpt-4', 'gpt-4-turbo')
            agent_type: Type of agent ('cloud', 'quantization', or 'hf')
        """
        self.agent_types = {
            "cloud": CloudAgent,
            "ollama": OllamaAgent,
        }

        self.agent = self._init_agent(agent_name, agent_type, **kwargs)

    def _init_agent(self, agent_name: str, agent_type: str, **kwargs) -> BaseAgent:
        agent_type = agent_type.lower()
        if agent_type not in self.agent_types:
            raise InvalidAgentType(
                f"Invalid agent type. Must be one of: {', '.join(self.agent_types.keys())}"
            )

        agent_class = self.agent_types[agent_type]
        return agent_class(agent_name, agent_type, **kwargs)

    def process(self,**kwargs) -> Any:
        return self.agent.process(**kwargs)

    def get_agent_info(self) -> Dict[str, str]:
        return {
            "name": self.agent.name,
            "type": type(self.agent).__name__,
            "config": getattr(self.agent, "model", None),
        }
