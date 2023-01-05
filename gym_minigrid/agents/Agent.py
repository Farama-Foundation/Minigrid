from typing import Tuple
import random
import numpy as np

class Agent:
    def __init__(
        self, 
        id,
        position: Tuple[int, int], 
        direction: int,
        DML: bool,
        p_exchg: float
        ):

        self.id = id

        prob = np.random.random()
        if prob < 0.05:
            self.maxSpeed = 2
        elif prob < 0.95:
            self.maxSpeed = 3
        else:
            self.maxSpeed = 4

        self.DML = DML
        self.p_exchg = p_exchg

        self.initPosition = position
        self.initDirection = direction

        self.position = position
        self.direction = direction
        self.speed = self.maxSpeed

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