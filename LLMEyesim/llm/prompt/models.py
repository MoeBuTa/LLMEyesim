from dataclasses import dataclass


@dataclass
class SystemPrompt:
    """Dataclass for system prompts."""
    role: str = "You are an executive agent in a mobile robotic system."
    environment: str = "a simulated indoor environment."
    mission: str = "Your mission is to navigate the robot to the red can."
    capabilities: str = """
You can move the robot in any of the eight directions by a distance of 100, 200, or 300 units per step. The directions are defined as follows: north (0°), northeast (45°), east (90°), southeast (135°), south (180°), southwest (225°), west (270°), and northwest (315°).
    """
    response: str = ""

    def format_system_prompt(self):
        return f"""
{self.role}
The robot is in {self.environment}.
Your mission is to {self.mission}.
Your robot has the following capabilities: {self.capabilities}.
You will receive the exploration records of the current environment, the robot's current position, and the action queue planned for the robot. 
Based on this information, generate a full action queue by updating the current action queue to avoid obstacles, optimize its performance, and ensure the mission is successfully completed. 
Present your decisions along with justifications for each action.
{self.response}
        """


@dataclass
class UserPrompt:
    """Dataclass for user prompts."""
    environment_information: str = "The obstacles in the environment are located at the following positions: (1, 1), (2, 2), (3, 3). The target location is at position (4, 4)."
    robot_state: str = "The robot is currently at position (0, 0). The robot currently has a queue of actions"

    def format_user_prompt(self):
        return f"""
Environment Information:        
{self.environment_information}
The Robot's Current State:
{self.robot_state}
        """


@dataclass
class EnvironmentInformation:
    """Dataclass for environment information."""
    exploration_records: str

    def format_environment_information(self):
        return f"""
        Currently the robot has found the following items: {self.exploration_records}
        """
