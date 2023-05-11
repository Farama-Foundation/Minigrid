from typing import Tuple
from gym_minigrid.lib.BaseObject import BaseObject
import random
import numpy as np

class Agent(BaseObject):
    def __init__(
        self, 
        id,
        position: Tuple[int, int], 
        direction: int, # TODO convert direction to enum
        maxSpeed: float = 4,
        speed: float = 3
        ):

        self.id = id

        super().__init__(
            initTopLeft=initTopLeft,
            initBottomRight=initBottomRight
        )

        self.initPosition = position
        self.initDirection = direction

        self.position = position
        self.direction = direction
        self.maxSpeed = maxSpeed
        self.speed = speed

        self.canShiftLeft = True
        self.canShiftRight = True
        self.agentAction = 0 # Put this in PedAgent later, save agentActions here for execution
        # Agents will save the actions we compute
        # Once we execute, we get action from here
        
        self.gapSame = 8
        self.gapOpp = 4
        
        pass

    def reset(self):
        # TODO 
        
        self.position = self.initPosition 
        self.direction = self.initDirection

        pass