class PromptV2:
    def __init__(self):
        pass

    @staticmethod
    def create_system_prompt(role_description: str="", environment_description: str="", mission_description: str="", capabilities_description: str="", response_description: str="") -> str:
        role = role_description if role_description else """You are an executive agent in a mobile robotic system. """

        environment = environment_description if environment_description else """a simulated 6000x6000 indoor environment."""

        mission = mission_description if mission_description else """navigate the robot to find and reach all targets in the environment."""

        capabilities = capabilities_description if capabilities_description else """You can move the robot in any of the four directions by a distance of 100, 200, or 300 units per step. The directions are defined as follows: north (0°), northeast (45°), east (90°)."""

        response = response_description if response_description else "Based on this information, generate a full action queue to keep the robot at least 200 units away from obstacles, find and reach all targets in the environment. Present your decisions along with justifications for each action."

        response_waypoint = response_description if response_description else "Based on this information, generate a list of waypoints to keep the robot at least 400 units away from obstacles, find and reach all targets in the environment."

        return f"""
{role}
The robot is in {environment}.
Your mission is to {mission}.
There are 4 targets in the environment. Located at top middle, left middle, right middle, and bottom middle.
A target is considered reached when the robot is within 100 units of the target location.
Your robot has the following capabilities: {capabilities}.
You will receive the robot state and exploration records as input.
{response_waypoint}"""


    @staticmethod
    def create_user_prompt(message: str) -> str:
        return f"""
{message}
        """


    @staticmethod
    def example_user_prompt() -> str:
        message = "The robot has found the following items: target - location at (4, 4) The robot is currently at position (0, 0). The robot currently has a queue of actions"
        return PromptV2.create_user_prompt(message)

    @staticmethod
    def example_assistant_prompt() -> str:
        return "The robot has successfully navigated to the target location"
