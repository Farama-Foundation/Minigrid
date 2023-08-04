from typing import Tuple, List
from pedgrid.lib.BaseObject import BaseObject

class Sidewalk(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        sidewalkID: int,
        objectType="Sidewalk"
    ):
        super().__init__(
            topLeft=topLeft,
            bottomRight=bottomRight,
            objectType=objectType
        )
        
        self.sidewalkID = sidewalkID