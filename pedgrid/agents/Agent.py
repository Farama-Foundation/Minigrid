from typing import Tuple
from pedgrid.lib.BaseObject import BaseObject
import random
import numpy as np
from pedgrid.lib.Action import Action
from abc import abstractmethod

class Agent(BaseObject):
    def __init__(
        self, 
        id,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        direction: int, # TODO convert direction to enum
        maxSpeed: float = 4,
        speed: float = 3,
        objectType="Agent"
    ):

        self.id = id

        super().__init__(
            topLeft=topLeft,
            bottomRight=bottomRight,
            objectType=objectType
        )

        self.initTopLeft = topLeft
        self.initBottomRight = bottomRight
        self.initDirection = direction

        self.direction = direction
        self.maxSpeed = maxSpeed
        self.speed = speed

        self.canShiftLeft = True
        self.canShiftRight = True
        self.agentAction = 0 # Put this in PedAgent later, save agentActions here for execution
        # Agents will save the actions we compute
        # Once we execute, we get action from here
        
        self.gapSame = 8 # TODO transfer all the blue and alder properties
        self.gapOpp = 4
        
        pass

    def reset(self):
        # TODO 
        self.topLeft = self.initTopLeft
        self.bottomRight = self.initBottomRight
        self.direction = self.initDirection
        pass

    @abstractmethod
    def parallel1(self, env) -> Action:
        raise NotImplementedError("parallel1 is not implemented")

    @abstractmethod
    def parallel2(self, env) -> Action:
        raise NotImplementedError("parallel2 is not implemented")