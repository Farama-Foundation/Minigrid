from enum import Enum
from telnetlib import FORWARD_X
from tkinter import MOVETO

class ObjectAction(Enum):
    """Object actions are relative to the world coordinate system

    Args:
        Enum (_type_): _description_
    """
    MOVETO = "MOVETO",
    FORWARD = "FORWARD"
