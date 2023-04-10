from typing import Tuple, List

class Object:
    def __init__(
        self,
        initTopLeft: Tuple[int, int],
        initBottomRight: Tuple[int, int]
    ):
        self.initTopLeft = initTopLeft
        self.initBottomRight = initBottomRight

        self.topLeft = initTopLeft
        self.bottomRight = initBottomRight

        self.length = self.bottomRight[0]-self.topLeft[0]
        self.width = self.bottomRight[1]-self.topLeft[1]
    
    def reset(self):
        self.topLeft = self.initTopLeft
        self.bottomRight = self.initBottomRight