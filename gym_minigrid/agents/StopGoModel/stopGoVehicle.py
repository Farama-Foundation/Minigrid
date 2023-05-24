from gym_minigrid.agents.Vehicle import Vehicle
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.VehicleAction import VehicleAction

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
        return Action(self, VehicleAction.KEEP)