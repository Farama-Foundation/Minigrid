from enum import Enum

class LaneAction(Enum):
    """Lane actions are relative to the world coordinate system

    Args:
        Enum (_type_): _description_
    """
    KEEP = "KEEP",
    LEFT = "LEFT",
    RIGHT = "RIGHT"