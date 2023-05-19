import numpy as np
from gym_minigrid.agents.Vehicle import Vehicle
from gym_minigrid.minigrid import Grid
from typing import List
from gym_minigrid.agents.PedAgent import PedAgent

class PedGrid(Grid):

    def render(
        self,
        tile_size,
        pedAgents: List[PedAgent], # need to add support for roads (lanes), sidewalks, vehicles
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None,
        vehicles: List[Vehicle]=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        self.pedAgents = pedAgents
        self.vehicles = vehicles

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                
                pedAgentIndex = self.getPedAgentInThisPosition(i, j)
                vehicleIndex = self.getVehicleInThisPosition(i, j)
                # if vehicleIndex is not None:
                #     print(f"Vehicle in position {i}, {j}")
                #     print(f'Vehicle top left: {self.vehicles[vehicleIndex].topLeft}')
                #     print(f'Vehicle bottom right: {self.vehicles[vehicleIndex].bottomRight}')
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=pedAgents[pedAgentIndex].direction if pedAgentIndex is not None else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    isVehicle= vehicleIndex is not None
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img
    
    def getVehicleInThisPosition(self, positionX, positionY):
        """
            Returns the vehicle index in the given position
            A vehicle is in a position if the position is in the vehicle's rectangle
        args:
            positionX: x coordinate of the position
            positionY: y coordinate of the position
        returns:
            vehicleIndex: index of the vehicle in the given position (none if there is no vehicle in the position)
        """
        if self.vehicles is None:
            return None
        for index, vehicle in enumerate(self.vehicles):
            if positionX >= vehicle.topLeft[0] and positionX <= vehicle.bottomRight[0] and positionY >= vehicle.topLeft[1] and positionY <= vehicle.bottomRight[1]:
                return index
        return None

    def getPedAgentInThisPosition(self, positionX, positionY):
        """
            Returns the pedAgent index in the given position, or None if there is no pedAgent in the position
        args:
            positionX: x coordinate of the position
            positionY: y coordinate of the position
        returns:
            pedAgentIndex: index of the pedAgent in the given position (none if there is no pedAgent in the position)
        """
       
        for index, pedAgent in enumerate(self.pedAgents):
            if np.array_equal(pedAgent.position, (positionX, positionY)):
                return index
            
        return None

        