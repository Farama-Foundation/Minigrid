from typing import Tuple, List
from gym_minigrid.agents.Agent import Agent

class Vehicle(Agent):
    def __init__(
        self,
        id,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        direction: int,
        maxSpeed: float,
        speed: float,
        inLane: int
    ):
        super.__init__(
            id=id,
            initTopLeft=topLeft,
            initBottomRight=bottomRight,
            direction=direction,
            maxSpeed=maxSpeed,
            speed=speed
        )
        
        self.inLine = inLane