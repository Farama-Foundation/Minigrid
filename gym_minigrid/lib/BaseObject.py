from typing import Tuple, List

class BaseObject:
    def __init__(
        self,
        initTopLeft: Tuple[int, int],
        initBottomRight: Tuple[int, int]
    ):
        self.initTopLeft = initTopLeft
        self.initBottomRight = initBottomRight

        self.topLeft = initTopLeft
        self.bottomRight = initBottomRight

        self.width = self.bottomRight[0]-self.topLeft[0] + 1
        self.height = self.bottomRight[1]-self.topLeft[1] + 1
    
    def reset(self):
        self.topLeft = self.initTopLeft
        self.bottomRight = self.initBottomRight
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.topLeft[0] + self.bottomRight[0]) // 2, (self.topLeft[1] + self.bottomRight[1]) // 2

    @property
    def position(self) -> Tuple[int, int]:
        return self.center