from typing import Tuple, List
from gym_minigrid.lib.BaseObject import BaseObject

class Vehicle(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int]
    ):
        super.__init__(
            topLeft=topLeft,
            bottomRight=bottomRight
        )