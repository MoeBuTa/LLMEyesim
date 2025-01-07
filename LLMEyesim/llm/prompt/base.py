

class Prompt:
    def __init__(self, system_prompt: str, user_prompt: str, example_user_prompt: str, example_assistant_prompt: str):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.example_user_prompt = example_user_prompt
        self.assistant_prompt = example_assistant_prompt

    def example_user_prompt(self):
        return f""""""

    @staticmethod
    def format_system_prompt(environment:str, mission: str, capabilities: str, response_format: str):

        return f"""
You are controlling a mobile robot. 
The robot is in {environment}. 
Your mission is to {mission}.
Your robot has the following capabilities: {capabilities}.
Based on the summary of robot's current state, make actionable decisions from the action dictionary to complete the goal.
Follow this JSON format to generate control signals and justifications: {response_format}
"""

    @staticmethod
    def format_user_prompt():
        return f""""""
