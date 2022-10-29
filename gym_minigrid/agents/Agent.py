from typing import Tuple
class Agent:
    def __init__(
        self, 
        position: Tuple[int, int], 
        direction: int, 
        speed = 2,
        ):
        self.position = position
        self.direction = direction
        self.speed = speed

        pass