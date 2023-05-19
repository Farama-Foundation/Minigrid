from typing import Tuple, List
from gym_minigrid.agents.Agent import Agent
from gym_minigrid.rendering import fill_coords, point_in_line, point_in_rect

class Vehicle(Agent):
    def __init__(
        self,
        id,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        direction: int,
        maxSpeed: float,
        speed: float,
        inRoad: int,
        inLane: int
    ):
        super().__init__(
            id=id,
            initTopLeft=topLeft,
            initBottomRight=bottomRight,
            direction=direction,
            maxSpeed=maxSpeed,
            speed=speed
        )
        
        self.inRoad = inRoad
        self.inLine = inLane

    