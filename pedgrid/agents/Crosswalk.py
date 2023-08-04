from typing import Tuple, List
from pedgrid.lib.BaseObject import BaseObject
from pedgrid.lib.Direction import Direction
from .Vehicle import Vehicle
import numpy as np

class Crosswalk(BaseObject):
    def __init__(
        self,
        topLeft: Tuple[int, int],
        bottomRight: Tuple[int, int],
        crosswalkID: int,
        overlapRoad: int,
        overlapLanes: List[int], # list of lane IDs
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
        # first reset previous incoming vehicles
        self.incomingVehicles = np.empty(len(self.overlapLanes), Vehicle)
        
        # calculate based on lane ID and direction
        vehicleAgents = env.vehicleAgents

        vehicleInSameLanes = [[] for _ in range(len(self.overlapLanes))]
        for vehicle in vehicleAgents:
            if vehicle.inLane in self.overlapLanes:
                vehicleInSameLanes[self.overlapLanes.index(vehicle.inLane)].append(vehicle)

        for i in range(0, len(self.overlapLanes)):
            lanesInRoad = env.road.lanes
            for lane in lanesInRoad:
                if lane.laneID == self.overlapLanes[i]:
                    laneDirection = lane.direction
                    break

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
