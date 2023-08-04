from typing import Tuple, List
from pedgrid.lib.BaseObject import BaseObject

class Building(BaseObject): # to add more variety of objects - Taorui
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        buildingID: int,
        height: int = None,
        objectType="Building"
    ):
        super().__init__(
            topLeft=topLeft,
            bottomRight=bottomRight,
            objectType=objectType
        )
        
        self.buildingID = buildingID
        self.height = height