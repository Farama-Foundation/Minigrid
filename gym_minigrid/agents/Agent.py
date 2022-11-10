from typing import Tuple

import random
class Agent:
    def __init__(
        self, 
        position: Tuple[int, int], 
        direction: int, 
        speed = 2,
        ):

        self.initPosition = position
        self.initDirection = direction
        self.initSpeed = speed

        self.position = position
        self.direction = direction
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
        self.speed = self.initSpeed

        pass

    def getAction(self):
        return random.randint(0, 2)