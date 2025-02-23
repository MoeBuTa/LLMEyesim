from typing import Dict, List

from openai.types.chat import completion_create_params

from LLMEyesim.llm.llm.manager import LLMManager
from LLMEyesim.llm.prompt.prompt_v1 import PromptV1
from LLMEyesim.llm.prompt.prompt_v2 import PromptV2


class ExecutiveAgent:
    def __init__(self, llm_name="gpt-4o", llm_type="cloud"):
        self.llm = LLMManager(llm_name, llm_type)
        self.llm_name = llm_name

    def process(self, images: List, human_instruction: str = None, last_command=None,
                enable_defence: bool = False) -> Dict:
        action_generation_prompt = PromptV1(enable_defence)
        system_prompt = action_generation_prompt.create_system_prompt()
        user_prompt = action_generation_prompt.create_user_prompt(images, human_instruction, last_command)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self.llm.process(messages=messages)

    def process_v2(self, message: str, response_format: completion_create_params.ResponseFormat,
                   prompt_type: int = 0) -> Dict:
        """
        Process the executive agent with the given exploration records and robot state.
        """
        system_prompt = PromptV2.create_system_prompt()
        user_prompt = PromptV2.create_user_prompt(message=message)
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        if prompt_type == '1':
            example_user_prompt = PromptV2.example_user_prompt()
            example_assistant_prompt = PromptV2.example_assistant_prompt()
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": example_user_prompt},
                        {"role": "assistant", "content": example_assistant_prompt},
                        {"role": "user", "content": user_prompt}]
        return self.llm.process_v2(messages=messages, response_format=response_format)
