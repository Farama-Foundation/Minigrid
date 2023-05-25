from typing import Tuple, List
from gym_minigrid.lib.BaseObject import BaseObject
from .Vehicle import Vehicle
import numpy as np

class Crosswalk(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        crosswalkID: int,
        overlapRoad: int,
        overlapLanes: List[int],
        objectType="Crosswalk"
    ):
        super().__init__(
            topLeft=topLeft,
            bottomRight=bottomRight,
            objectType=objectType
        )
        
        self.crosswalkID = crosswalkID
        self.overlapRoad = overlapRoad
        self.overlapLanes = overlapLanes
        
        self.lastVehiclesCrossed = np.empty(len(overlapLanes), Vehicle)