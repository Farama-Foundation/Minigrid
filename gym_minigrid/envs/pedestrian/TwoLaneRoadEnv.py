from .PedestrianEnv import PedestrianEnv

from typing import List
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import *
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
        pedAgents: List[PedAgent]=None,
        vehicleAgents: List[Vehicle]=None,
        road: Road=None, # TODO Color code roads, sidewalks
        sidewalks: List[Sidewalk]=None,
        width=8,
        height=8,
        stepsIgnore = 100
    ):
        
        super().__init__(
            pedAgents=pedAgents,
            width=width,
            height=height,
            stepsIgnore=stepsIgnore
        )

        self.vehicleAgents = vehicleAgents
        self.road = road
        self.sidewalks = sidewalks

        # TODO label each tile with either lane/sidewalk?

        pass

    def getVehicleAgents(self):
        return self.vehicleAgents
    
    def addVehicleAgents(self, agents: List[Vehicle]):
        for agent in agents:
            self.addVehicleAgent(agent)
    
    def addVehicleAgent(self, agent: Vehicle):
        self.vehicleAgents.append(agent)
        # subscribe to events here

    def getNumVehicleAgents(self):
        return len(self.vehicleAgents)
    
    def resetAgents(self):
        for agent in self.vehicleAgents:
            agent.reset()
    
    def removeVehicleAgent(self, agent: Vehicle):
        if agent in self.vehicleAgents:
            # unsubscribe to events here
            self.vehicleAgents.remove(agent)
        else:
            logging.warn("Agent not in list")
    
    def forwardVehicle(self, agent: Vehicle):
        assert agent.direction >= 0 and agent.direction < 4
        fwd_pos = agent.topLeft + agent.speed * DIR_TO_VEC[agent.direction]
        if fwd_pos[0] < 0 or fwd_pos[0] + agent.width >= self.width \
            or fwd_pos[1] < 0 or fwd_pos[1] + agent.height >= self.height:
            logging.warn("Vehicle cannot be moved here - out of bounds")
        
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward if no overlap
        if fwd_cell == None or fwd_cell.can_overlap():
            agent.topLeft = fwd_pos
            agent.bottomRight = (fwd_pos[0]+agent.width, fwd_pos[1]+agent.height)

register(
    id='TwoLaneRoadEnv-20x80-v0',
    entry_point='gym_minigrid.envs.pedestrian.PedestrianEnv:TwoLaneRoadEnv'
)