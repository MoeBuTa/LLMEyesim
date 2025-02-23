from typing import List

from pydantic import BaseModel, ConfigDict, Field


class ActionQueue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class RobotAction(BaseModel):
        model_config = ConfigDict(extra="forbid")
        direction: str = Field(..., description="The direction of the action")
        distance: int = Field(..., description="The distance of the action")
        justification: str = Field(..., description="Justification for the action")


    action_queue: List[RobotAction] = Field(..., description="List of actions to be executed")


class WayPointList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class WayPoint(BaseModel):
        model_config = ConfigDict(extra="forbid")
        x: int = Field(..., description="The x-coordinate of the waypoint")
        y: int = Field(..., description="The y-coordinate of the waypoint")
        description: str = Field(..., description="Description of the waypoint")
    waypoint_list: List[WayPoint] = Field(..., description="List of waypoints")