from dataclasses import dataclass

@dataclass
class SystemPrompt:
    """Dataclass for system prompts."""
    role: str = "You are an executive agent in a mobile robotic system."
    environment: str = "The robot is in a simulated environment."
    mission: str = "Your mission is to navigate to the target location."
    capabilities: str = "The robot has the following capabilities: perception, planning, and control."
    response: str = "Follow this JSON format to generate control signals and justifications: {response_format}"


@dataclass
class UserPrompt:
    """Dataclass for user prompts."""
