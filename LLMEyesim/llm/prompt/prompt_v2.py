from LLMEyesim.llm.prompt.models import EnvironmentInformation, SystemPrompt, UserPrompt


class PromptV2:
    def __init__(self):
        pass

    @staticmethod
    def create_system_prompt() -> str:
        environment = "a simulated indoor environment."
        mission = "Your mission is to navigate the robot to the red can."
        response = ""
        system_prompt = SystemPrompt(environment=environment, mission=mission, response=response)
        return system_prompt.format_system_prompt()

    @staticmethod
    def create_user_prompt(exploration_records: str, robot_state: str) -> str:
        information = EnvironmentInformation(exploration_records=exploration_records)
        environment_information =  information.format_environment_information()
        user_prompt = UserPrompt(robot_state=robot_state, environment_information=environment_information)
        return user_prompt.format_user_prompt()

    @staticmethod
    def example_user_prompt() -> str:
        exploration_records = "The robot has found the following items: target - location at (4, 4)"
        robot_state = "The robot is currently at position (0, 0). The robot currently has a queue of actions"
        return PromptV2.create_user_prompt(exploration_records, robot_state)

    @staticmethod
    def example_assistant_prompt() -> str:
        return "The robot has successfully navigated to the target location"
