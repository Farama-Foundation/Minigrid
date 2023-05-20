from typing import Tuple, List
from .ObjectColors import ObjectColors
from gym_minigrid.rendering import *

class BaseObject:
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        objectType: str
    ):
        self.initTopLeft = topLeft
        self.initBottomRight = bottomRight

        self.topLeft = topLeft
        self.bottomRight = bottomRight

        self.width = self.bottomRight[0]-self.topLeft[0] + 1
        self.height = self.bottomRight[1]-self.topLeft[1] + 1

        self.objectType = objectType
        if self.objectType in ObjectColors.OBJECT_TO_IDX:
            self.color = ObjectColors.IDX_TO_COLOR[ObjectColors.OBJECT_TO_IDX[self.objectType]]
        else:
            self.color = None

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), ObjectColors.COLORS[self.color])

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (ObjectColors.OBJECT_TO_IDX[self.objectType], ObjectColors.COLOR_TO_IDX[self.color], 0)
    
    def reset(self):
        self.topLeft = self.initTopLeft
        self.bottomRight = self.initBottomRight
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.topLeft[0] + self.bottomRight[0]) // 2, (self.topLeft[1] + self.bottomRight[1]) // 2

    @property
    def position(self) -> Tuple[int, int]:
        return self.center