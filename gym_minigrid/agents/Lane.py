from typing import Tuple, List
from gym_minigrid.agents.Object import Object

class Lane(Object):
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