from dataclasses import dataclass


@dataclass
class SystemPrompt:
    """Dataclass for system prompts."""
    role: str = "You are an executive agent in a mobile robotic system."
    environment: str = "a simulated environment."
    mission: str = "Your mission is to navigate to the target location."
    capabilities: str = """
    You can move the robot in eight directions at a time: north, south, east, west, northeast, northwest, southeast, and southwest.
    """
    response: str = ""

    def format_system_prompt(self):
        return f"""
        {self.role}
        The robot is in {self.environment}.
        Your mission is to {self.mission}.
        Your robot has the following capabilities: {self.capabilities}.
        Based on the summary of robot's current state, make actionable decisions from the capabilities to complete the goal.
        Follow this JSON format to generate your decisions and justifications: 
        {self.response}
        """


@dataclass
class UserPrompt:
    """Dataclass for user prompts."""
    environment_information: str = "The obstacles in the environment are located at the following positions: (1, 1), (2, 2), (3, 3). The target location is at position (4, 4)."
    current_position: str = "The robot is currently at position (0, 0)."
    action_queue: str = "The robot currently has a queue of actions"

    def format_user_prompt(self):
        return f"""
         {self.environment_information}
        {self.current_position}
        {self.action_queue}
        """


@dataclass
class EnvironmentInformation:
    """Dataclass for environment information."""
    located_obstacles: str
    located_target: str

    def format_environment_information(self):
        return f"""
        The obstacles in the environment are located at the following positions: {self.located_obstacles}.
        The target location is at position {self.located_target}.
        """
