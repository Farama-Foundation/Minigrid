from typing import Tuple, List
from gym_minigrid.agents.BaseBaseBaseObject import BaseObject

class Lane(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        direction: int,
        laneID: int
    ):
        super.__init__(
            topLeft=topLeft,
            bottomRight=bottomRight
        )
        self.direction = direction
        self.laneID = laneID