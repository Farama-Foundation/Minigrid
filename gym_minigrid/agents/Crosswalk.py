from typing import Tuple, List
from gym_minigrid.lib.BaseObject import BaseObject
from gym_minigrid.lib.Direction import Direction
from .Vehicle import Vehicle
import numpy as np

class Crosswalk(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        crosswalkID: int,
        overlapRoad: int,
        overlapLanes: List[int],
        objectType="Crosswalk"
    ):
        super().__init__(
            topLeft=topLeft,
            bottomRight=bottomRight,
            objectType=objectType
        )
        
        self.crosswalkID = crosswalkID
        self.overlapRoad = overlapRoad
        self.overlapLanes = overlapLanes
        
        self.lastVehiclesCrossed = np.empty(len(self.overlapLanes), Vehicle)
        self.incomingVehicles = np.empty(len(self.overlapLanes), Vehicle)

    def updateIncomingVehicles(self, env):
        # calculate based on lane ID and direction
        vehicleAgents = env.vehicleAgents

        vehicleInSameLanes = np.empty(len(self.overlapLanes), List[Vehicle])
        for vehicle in vehicleAgents:
            if vehicle.inLane in self.overlapLanes:
                vehicleInSameLanes[self.overlapLanes.index(vehicle.inLane)].append(vehicle)

        for i in range(0, len(self.overlapLanes)):
            lanesInRoad = self.overlapRoad.lanes
            for lane in lanesInRoad:
                if lane.id == self.overlapLanes[i]:
                    laneDirection = lane.direction

            if laneDirection == Direction.North:
                for vehicle in vehicleInSameLanes[i]:
                    if vehicle.topLeft[1] > self.bottomRight[1] and (self.incomingVehicles[i] == None or vehicle.topLeft[1] < self.incomingVehicles[i].topLeft[1]):
                        self.incomingVehicles[i] = vehicle
            elif laneDirection == Direction.South:
                for vehicle in vehicleInSameLanes[i]:
                    if vehicle.topLeft[1] < self.bottomRight[1] and (self.incomingVehicles[i] == None or vehicle.topLeft[1] > self.incomingVehicles[i].topLeft[1]):
                        self.incomingVehicles[i] = vehicle
            elif laneDirection == Direction.West:
                for vehicle in vehicleInSameLanes[i]:
                    if vehicle.topLeft[0] < self.bottomRight[0] and (self.incomingVehicles[i] == None or vehicle.topLeft[0] > self.incomingVehicles[i].topLeft[0]):
                        self.incomingVehicles[i] = vehicle
            elif laneDirection == Direction.East:
                for vehicle in vehicleInSameLanes[i]:
                    if vehicle.topLeft[0] > self.bottomRight[0] and (self.incomingVehicles[i] == None or vehicle.topLeft[0] < self.incomingVehicles[i].topLeft[0]):
                        self.incomingVehicles[i] = vehicle
