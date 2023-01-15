from gym_minigrid.agents.Agent import Agent
from gym_minigrid.agents import Lanes
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.ForwardAction import ForwardAction
import numpy as np
import logging
from typing import Tuple

class PedAgent(Agent):

    def __init__(
        self, 
        id,
        position: Tuple[int, int], 
        direction: int, # TODO convert direction to enum,
        maxSpeed: float = 4,
        speed: float = 3,
        DML: bool = False, # TODO this is not a property of the agent.
        p_exchg: float = 0.0
        ):
    
        super().__init__(
            id=id,
            position=position,
            direction=direction,
            maxSpeed=maxSpeed,
            speed=speed,
            DML=DML,
            p_exchg=p_exchg
        )
