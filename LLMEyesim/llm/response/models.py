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