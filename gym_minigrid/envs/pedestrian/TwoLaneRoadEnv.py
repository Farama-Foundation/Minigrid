from .PedestrianEnv import PedestrianEnv

from typing import List
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import Agent, PedActions, PedAgent, Road, Lane
from gym_minigrid.envs.pedestrian.PedGrid import PedGrid
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.ForwardAction import ForwardAction
from gym_minigrid.lib.Direction import Direction
from .EnvEvent import EnvEvent
import logging
import random

class TwoLaneRoadEnv(PedestrianEnv):
    # Write the outline here how it should work

    # generic object representation
    # generic actors?
    def __init__(
        self,
        agents: List[Agent]=None,
        roads: List[Road]=None,
        width=8,
        height=8,
        stepsIgnore = 100
    ):
        
        super().__init__(
            agents=agents,
            width=width,
            height=height,
            stepsIgnore=stepsIgnore
        )
        

    pass


register(
    id='TwoLaneRoadEnv-20x80-v0',
    entry_point='gym_minigrid.envs.pedestrian.MultiPedestrianEnv:TwoLaneRoadEnv'
)