from typing import Tuple, List
from gym_minigrid.lib.BaseObject import BaseObject

class Lane(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        direction: int,
        laneID: int,
        posRelativeToCenter: int
        # ^ we assume roads are divided at center
        # with offsets, one side positive, other negative
        # ex. +2, +1, -1, -2 can be positions w/ center 0
        # TODO how will 3 lanes work?
        # +2, +1, -1 for 2 lanes on one side and 1 lane on other???
    ):
        super.__init__(
            topLeft=topLeft,
            bottomRight=bottomRight
        )
        self.direction = direction
        self.laneID = laneID
        self.posRelativeTOCenter = posRelativeToCenter