from typing import Tuple, List
from gym_minigrid.lib.BaseObject import BaseObject
from gym_minigrid.agents.Lane import Lane

class Road(BaseObject):
    def __init__(
        self,
        lanes: List[Lane],
        roadID: int
    ):
        self.lanes = lanes
        self.lanes.sort(key=lambda lane: lane.topLeft)
        if lanes.len() != 0:
            self.topLeft = lanes[0].topLeft
            self.bottomRight = lanes[lanes.len()-1].bottomRight
        else:
            self.topLeft = None
            self.bottomRight = None

        self.roadID = roadID

        super.__init__(
            topLeft=self.topLeft,
            bottomRight=self.bottomRight
        )