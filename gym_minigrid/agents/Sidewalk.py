from typing import Tuple, List
from gym_minigrid.lib.BaseObject import BaseObject

class Sidewalk(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        sidewalkID: int
    ):
        super.__init__(
            topLeft=topLeft,
            bottomRight=bottomRight
        )
        self.sidewalkID = sidewalkID