import logging
from typing import Tuple, List

import numpy as np
import math

from pedgrid.agents import LaneNum
from pedgrid.agents.PedAgent import PedAgent


from pedgrid.lib.LaneAction import LaneAction
from pedgrid.lib.Action import Action
from pedgrid.lib.Direction import Direction
import logging

from pedgrid.lib.ObjectAction import ObjectAction

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
        # logging.warn(f"pedestrian {self.id} has planned {self.stepsPlanned} steps")
        if self.stepsPlanned == 0:
            # won't work now : finish distanceBetweenTwoVehicles
            print(f"Distance between two vehicles for ped {self.id}: {self.timeBetweenTwoVehicles(env)}")
            if self.timeBetweenTwoVehicles(env) > self.minTimeToCross:
                self.stepsPlanned = env.width / (self.speed * 2)
                logging.warn("pedestrian {} planned {} steps".format(self.id, self.stepsPlanned))
        # Calculated the whether the agent should stop or go 
        # if it has remaining steps to perform, don't do anything
            pass


    def parallel2(self, env): # TODO add type
        if self.stepsPlanned > 0:
            self.stepsPlanned -= 1
            return Action(self, ObjectAction.FORWARD)
        return None
        
    def timeBetweenTwoVehicles(self, env):
        crosswalks = env.crosswalks
        closestCrosswalk = None # closest crosswalk in the correct direction
        closestDist = math.inf
        for crosswalk in crosswalks:
            if (self.direction == Direction.East and crosswalk.bottomRight[0] > self.position[0]) \
                    or (self.direction == Direction.West and crosswalk.topLeft[0] < self.position[0]) \
                    or (self.direction == Direction.North and crosswalk.topLeft[1] < self.position[1]) \
                    or (self.direction == Direction.South and crosswalk.bottomRight[1] > self.position[1]):
                newDist = (self.position[0] - crosswalk.center[0])**2 + (self.position[1] - crosswalk.center[1])**2
                if newDist < closestDist:
                        closestCrosswalk = crosswalk
                        closestDist = newDist

        laneIDs = closestCrosswalk.overlapLanes
        lanes = env.road.lanes
        overlapLanes = []
        for lane in lanes:
            if lane.laneID in laneIDs:
                overlapLanes.append(lane)
        closestLane = None
        closestDist = math.inf
        for lane in overlapLanes:
            newDist = (self.position[0] - lane.center[0])**2 + (self.position[1] - lane.center[1])**2
            if newDist < closestDist:
                closestLane = lane
                closestDist = newDist
        
        # TO-DO : We need to find out the vehicle in the crosswalk given the lane of ped
        # crossWalkVehicle = env.getVehicleInCrosswalk(self.inLane)
        # TO-DO : We need to find the closest incoming vehicle in the lane of ped
        # Lane ID affects which side of crosswalk we need to look at
        closestCrosswalk.updateIncomingVehicles(env)
        laneIndex = closestCrosswalk.overlapLanes.index(closestLane.laneID)
        incomingVehicle = closestCrosswalk.incomingVehicles[laneIndex]
        if incomingVehicle == None:
            return math.inf

        # if crossWalkVehicle == None:
            # Find the distance between incoming and middle of crosswalk
            # if self.direction == Direction.East:
            #     return abs(incomingVehicle.topLeft[1] - env.crosswalks[0].topLeft[1])
            # elif self.direction == Direction.West:
            #     return abs(incomingVehicle.bottomRight[1] - env.crosswalks[0].bottomRight[1])
        
        if self.direction == Direction.East:
            return abs(incomingVehicle.topLeft[1] - closestCrosswalk.topLeft[1]) / incomingVehicle.speed
        elif self.direction == Direction.West:
            return abs(incomingVehicle.bottomRight[1] - closestCrosswalk.bottomRight[1]) / incomingVehicle.speed
        
