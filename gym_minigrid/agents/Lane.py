from typing import Tuple, List
from gym_minigrid.lib.BaseObject import BaseObject
from gym_minigrid.lib.ObjectColors import ObjectColors
from gym_minigrid.rendering import *

class Lane(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        direction: int,
        inRoad: int,
        laneID: int,
        posRelativeToCenter: int,
        # ^ we assume roads are divided at center
        # with offsets, one side positive, other negative
        # ex. +2, +1, -1, -2 can be positions w/ center 0
        # TODO how will 3 lanes work?
        # +2, +1, -1 for 2 lanes on one side and 1 lane on other???
        objectType="Lane"
    ):
        super().__init__(
            topLeft=topLeft,
            bottomRight=bottomRight,
            objectType=objectType
        )

        self.direction = direction
        self.inRoad = inRoad
        self.laneID = laneID
        self.posRelativeTOCenter = posRelativeToCenter

    # def render(self, img, position):
    #     if position[0] == self.topLeft[0] or position[0] == self.bottomRight[0] \
    #     or position[1] == self.topLeft[1] or position[1] == self.bottomRight[1]:
    #         fill_coords(img, point_in_rect(0, 1, 0, 1), ObjectColors.COLORS[self.color])
    #     else:
    #         fill_coords(img, point_in_rect(0.3, 0.7, 0.3, 0.7), ObjectColors.COLORS[self.color])
    # ^ does not work, caching in rendering causes this to fail