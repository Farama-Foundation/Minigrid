import logging
from typing import Tuple, List

import numpy as np
import math

from gym_minigrid.agents import LaneNum
from gym_minigrid.agents.PedAgent import PedAgent


from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.ForwardAction import ForwardAction
from gym_minigrid.lib.Direction import Direction


class StopGoPed(PedAgent):

    def __init__(
        self, 
        id,
        position: Tuple[int, int], 
        direction: Direction, # TODO convert direction to enum,
        minTimeToCross: int,
        maxSpeed: float = 4,
        speed: float = 3,
        ):
    
        super().__init__(
            id=id,
            position=position,
            direction=direction,
            maxSpeed=maxSpeed,
            speed=speed
        )


        # number steps planned ahead without stopping
        self.stepsPlanned = 0
        self.minTimeToCross = minTimeToCross

    def parallel1(self, env): # TODO add type
        if self.stepsPlanned == 0:
            # won't work now : finish distanceBetweenTwoVehicles
            # if self.distanceBetweenTwoVehicles(env) > self.minTimeToCross:
                # self.stepsPlanned = env.width / self.speed
        # Calculated the whether the agent should stop or go 
        # if it has remaining steps to perform, don't do anything
            pass


    def parallel2(self, env): # TODO add type
        if self.stepsPlanned > 0:
            self.stepsPlanned -= 1
            return Action(self, ForwardAction.KEEP)
        return None
        
    def distanceBetweenTwoVehicles(self, env):
        # TO-DO : We need to find out the vehicle in the crosswalk given the lane of ped
        crossWalkVehicle = env.getVehicleInCrosswalk(self.inLane)
        # TO-DO : We need to find the closest incoming vehicle in the lane of ped
        # Lane ID affects which side of crosswalk we need to look at
        env.crosswalks[0].updateIncomingVehicles()
        laneIndex = env.crosswalks[0].overlapLanes.index(self.inLane)
        incomingVehicle = env.crosswalks[0].incomingVehicles[laneIndex]
        if incomingVehicle == None:
            return math.inf
        

        if crossWalkVehicle == None:
            # Find the distance between incoming and middle of crosswalk
            if self.direction == Direction.East:
                return abs(incomingVehicle.topLeft[1] - env.crosswalks[0].topLeft[1])
            elif self.direction == Direction.West:
                return abs(incomingVehicle.bottomRight[1] - env.crosswalks[0].bottomRight[1])
        
