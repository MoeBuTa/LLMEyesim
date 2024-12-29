from typing import Dict, List

from LLMEyeSim.llm.agents.manager import AgentManager
from LLMEyeSim.llm.prompt.action_generation import (
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

    def llm_response_record(
        self,
        step: int,
        perception: str,
        planning: str,
        control: List[Dict],
        attack_injected: bool,
        completion_tokens: int,
        prompt_tokens: int,
        total_tokens: int,
        response_time: float

    ):
        return {
            "step": step,
            "task_name": self.task_name,
            "model_name": self.agent_name,
            "perception": perception,
            "planning": planning,
            "control": control,
            "attack_injected": attack_injected,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "response_time": response_time
        }
