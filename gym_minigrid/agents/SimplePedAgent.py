import logging
from typing import Tuple, List

import numpy as np
import math

from gym_minigrid.agents import LaneNum

from .PedAgent import PedAgent
from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.ForwardAction import ForwardAction
from gym_minigrid.lib.Direction import Direction

class SimplePedAgent(PedAgent):
    """
    A simple pedestrian that moves forward
    """
    def __init__(self, id, position, direction, maxSpeed, speed):
        super().__init__(id, position, direction, maxSpeed, speed)
    
    def parallel1(self, env):
        """
            Simply move forward
        """
        return Action(self, ForwardAction.KEEP)
    
    def parallel2(self, env):
        pass