from typing import Tuple, List
from gym_minigrid.lib.BaseObject import BaseObject

class Crosswalk(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        crosswalkID: int,
        inRoad: int,
        inLanes: List[int],
        objectType="Crosswalk"
    ):
        super().__init__(
            topLeft=topLeft,
            bottomRight=bottomRight,
            objectType=objectType
        )
        
        self.crosswalkID = crosswalkID
        self.inRoad = inRoad
        self.inLanes = inLanes