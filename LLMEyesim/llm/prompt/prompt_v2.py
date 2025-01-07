from LLMEyesim.llm.prompt.models import SystemPrompt, UserPrompt, EnvironmentInformation


class PromptV2:
    def __init__(self):
        pass

    @staticmethod
    def create_system_prompt() -> str:
        environment = "a simulated environment."
        mission = "Your mission is to navigate to the target location."
        response = ""
        system_prompt = SystemPrompt(environment=environment, mission=mission, response=response)
        return system_prompt.format_system_prompt()

    @staticmethod
    def create_user_prompt(located_obstacles: str, located_target: str, current_position: str, action_queue: str) -> str:
        information = EnvironmentInformation(located_obstacles=located_obstacles, located_target=located_target)
        environment_information =  information.format_environment_information()
        user_prompt = UserPrompt(current_position=current_position, environment_information=environment_information, action_queue=action_queue)
        return user_prompt.format_user_prompt()

    @staticmethod
    def example_user_prompt() -> str:
        located_obstacles = "(1, 1), (2, 2), (3, 3)"
        located_target = "(4, 4)"
        current_position = "The robot is currently at position (0, 0)."
        action_queue = "The robot currently has a queue of actions"
        return PromptV2.create_user_prompt(located_obstacles, located_target, current_position, action_queue)

    @staticmethod
    def example_assistant_prompt() -> str:
        return "The robot has successfully navigated to the target location"
