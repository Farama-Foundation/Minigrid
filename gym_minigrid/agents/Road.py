from typing import Tuple, List
from gym_minigrid.agents.Object import Object
from gym_minigrid.agents.Lane import Lane

class Road(Object):
    def __init__(
        self,
        roads: List[Lane]
    ):
        self.roads = roads
        self.roads.sort(key=lambda lane: lane.topLeft)
        if roads.len() != 0:
            self.topLeft = roads[0].topLeft
            self.bottomRight = roads[roads.len()-1].bottomRight
        else:
            self.topLeft = None
            self.bottomRight = None

        super.__init__(
            topLeft=self.topLeft,
            bottomRight=self.bottomRight
        )