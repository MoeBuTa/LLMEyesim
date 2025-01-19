class PromptV2:
    def __init__(self):
        pass

    @staticmethod
    def create_system_prompt(role_description: str="", environment_description: str="", mission_description: str="", capabilities_description: str="", response_description: str="") -> str:
        role = role_description if role_description else """You are an executive agent in a mobile robotic system. """

        environment = environment_description if environment_description else """a simulated 2000x2000 indoor environment."""

        mission = mission_description if mission_description else """navigate the robot to all targets in the environment."""

        capabilities = capabilities_description if capabilities_description else """You can move the robot in any of the eight directions by a distance of 100, 200, or 300 units per step. The directions are defined as follows: north (0°), northeast (45°), east (90°), southeast (135°), south (180°), southwest (225°), west (270°), and northwest (315°)."""

        response = response_description if response_description else ""
        return f"""
{role}
The robot is in {environment}.
Your mission is to {mission}.
Your robot has the following capabilities: {capabilities}.
You will receive the exploration records collected by the robot, the robot's current position in x,y, where x represents west (smaller) and east (larger), and y represents south (smaller) and north (larger), and the action queue planned for the robot. 
Based on this information, generate a full action queue by updating the current action queue to avoid obstacles, keep the robot at least 200 units away from obstacles, optimize its performance, and ensure the mission is successfully completed. 
Present your decisions along with justifications for each action.
{response}"""


    @staticmethod
    def create_user_prompt(exploration_records_description: str="", robot_state_description: str="") -> str:

        exploration_records = exploration_records_description if exploration_records_description else ""

        robot_state = robot_state_description if robot_state_description else ""

        return f"""
{exploration_records}
{robot_state}
        """


    @staticmethod
    def example_user_prompt() -> str:
        exploration_records = "The robot has found the following items: target - location at (4, 4)"
        robot_state = "The robot is currently at position (0, 0). The robot currently has a queue of actions"
        return PromptV2.create_user_prompt(exploration_records, robot_state)

    @staticmethod
    def example_assistant_prompt() -> str:
        return "The robot has successfully navigated to the target location"
