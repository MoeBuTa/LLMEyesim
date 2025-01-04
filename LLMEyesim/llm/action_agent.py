from typing import Dict, List

from LLMEyesim.llm.agents.manager import AgentManager
from LLMEyesim.llm.prompt.action_generation_prompt_old import ActionGenerationPromptOld


class ActionAgent:
    def __init__(self, task_name:str, agent_name="gpt-4o", agent_type="cloud"):
        self.agent = AgentManager(agent_name, agent_type)
        self.task_name = task_name
        self.agent_name = agent_name

    def process_action(self, images: List, human_instruction: str=None, last_command=None, enable_defence: bool=False) -> Dict:
        action_generation_prompt_old = ActionGenerationPromptOld(enable_defence)
        system_prompt = action_generation_prompt_old.format_system_prompt()
        user_prompt = action_generation_prompt_old.format_user_prompt(images, human_instruction, last_command)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.agent.process(messages=messages)
