from typing import Tuple
import random

class Agent:
    def __init__(
        self, 
        position: Tuple[int, int], 
        direction: int, 
        speed = 1,
        ):

        self.initPosition = position
        self.initDirection = direction
        self.initSpeed = speed

        self.position = position
        self.direction = direction
        self.speed = speed

        pass

    def reset(self):
        # TODO 
        
        self.position = self.initPosition 
        self.direction = self.initDirection
        self.speed = self.initSpeed

        pass

    def getAction(self):
        return random.randint(0, 2)