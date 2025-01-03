from typing import Dict, List

from LLMEyesim.llm.agents.manager import AgentManager
from LLMEyesim.llm.prompt.action_generation import (
    action_system_prompt,
    action_user_prompt,
)


class ActionAgent:
    def __init__(self, task_name:str, agent_name="gpt-4o", agent_type="cloud"):
        self.agent = AgentManager(agent_name, agent_type)
        self.task_name = task_name
        self.agent_name = agent_name

    def process_action(self, images: List, human_instruction: str=None, last_command=None, enable_defence: bool=False) -> Dict:
        system_prompt = action_system_prompt(enable_defence)
        user_prompt = action_user_prompt(images, human_instruction, last_command)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.agent.process(messages=messages)
