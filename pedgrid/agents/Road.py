from typing import Tuple, List
from pedgrid.lib.BaseObject import BaseObject
from pedgrid.agents.Lane import Lane

class Road(BaseObject):
    def __init__(
        self,
        lanes: List[Lane],
        roadID: int,
        objectType="Road"
    ):
        self.lanes = lanes
        self.lanes.sort(key=lambda lane: lane.topLeft)
        if len(lanes) != 0:
            self.topLeft = lanes[0].topLeft
            self.bottomRight = lanes[len(lanes)-1].bottomRight
        else:
            self.topLeft = None
            self.bottomRight = None

        self.roadID = roadID

        super().__init__(
            topLeft=self.topLeft,
            bottomRight=self.bottomRight,
            objectType=objectType
        )

    # idea: add intersections in the future, either in this class or a separate one