from typing import Dict, List

from LLMEyesim.llm.api.manager import LLMManager
from LLMEyesim.llm.prompt.multi_modal_prompt import MultiModalPrompt


class ExecutiveAgent:
    def __init__(self, task_name:str, llm_name="gpt-4o", llm_type="cloud"):
        self.llm = LLMManager(llm_name, llm_type)
        self.task_name = task_name
        self.llm_name = llm_name

    def process(self, images: List, human_instruction: str=None, last_command=None, enable_defence: bool=False) -> Dict:
        action_generation_prompt = MultiModalPrompt(enable_defence)
        system_prompt = action_generation_prompt.format_system_prompt()
        user_prompt = action_generation_prompt.format_user_prompt(images, human_instruction, last_command)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.llm.process(messages=messages)
    
    def process_v2(self):
        pass