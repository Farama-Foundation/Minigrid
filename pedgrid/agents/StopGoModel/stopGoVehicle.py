from pedgrid.agents.Vehicle import Vehicle
from pedgrid.lib.Action import Action
from pedgrid.lib.VehicleAction import VehicleAction
from pedgrid.lib.Direction import Direction
import logging

class StopGoVehicle(Vehicle):
    """
    A simple vehicle that moves forward
    """
    def __init__(self, id, topLeft, bottomRight, direction, maxSpeed, speed, inRoad, inLane):
        super().__init__(id, topLeft, bottomRight, direction, maxSpeed, speed, inRoad, inLane)
    
    def go(self, env):
        """
            Simply move forward
        """
        self.followFrontVehicleSpeed(env)
        return Action(self, VehicleAction.KEEP)
    
    def followFrontVehicleSpeed(self, env):

        vehicleAgents = env.vehicleAgents

        vehicleInSameLane = []
        for vehicle in vehicleAgents:
            if (self.id == vehicle.id):
                continue
            elif self.inLane == vehicle.inLane:
                vehicleInSameLane.append(vehicle)

        vehicleInFront = None
        if self.direction == Direction.North:
            for vehicle in vehicleInSameLane:
                if vehicle.topLeft[1] < self.topLeft[1] and (vehicleInFront == None or vehicle.topLeft[1] > vehicleInFront.topLeft[1]):
                    vehicleInFront = vehicle
        elif self.direction == Direction.South:
            for vehicle in vehicleInSameLane:
                if vehicle.topLeft[1] > self.topLeft[1] and (vehicleInFront == None or vehicle.topLeft[1] < vehicleInFront.topLeft[1]):
                    vehicleInFront = vehicle
        elif self.direction == Direction.West:
            for vehicle in vehicleInSameLane:
                if vehicle.topLeft[0] < self.topLeft[0] and (vehicleInFront == None or vehicle.topLeft[0] > vehicleInFront.topLeft[0]):
                    vehicleInFront = vehicle
        elif self.direction == Direction.East:
            for vehicle in vehicleInSameLane:
                if vehicle.topLeft[0] > self.topLeft[0] and (vehicleInFront == None or vehicle.topLeft[0] < vehicleInFront.topLeft[0]):
                    vehicleInFront = vehicle

        if vehicleInFront == None:
            logging.warn("No vehicle in front")
        else:
            self.speed = vehicleInFront.speed
