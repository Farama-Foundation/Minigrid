import numpy as np
from gym_minigrid.minigrid import Grid, TILE_PIXELS
from typing import List
from gym_minigrid.agents import *
from gym_minigrid.rendering import *

class PedGrid(Grid):
    
    def render(
        self,
        tile_size,
        pedAgents: List[PedAgent]=[],
        vehicleAgents: List[Vehicle]=[],
        roads: List[Road]=[],
        sidewalks: List[Sidewalk]=[],
        crosswalks: List[Crosswalk]=[],
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Reset the grid
        self.grid = [None] * self.width * self.height

        # Fill/set grid with objects
        if len(roads) != 0:
            for road in roads:
                if road == None:
                    continue
                for lane in road.lanes:
                    # for x in range(lane.topLeft[0], lane.bottomRight[0]):
                    #     for y in range(lane.topLeft[1], lane.bottomRight[1]):
                    #         self.set(x, y, lane)
                    for x in range(lane.topLeft[0], lane.bottomRight[0]+1):
                        self.set(x, lane.topLeft[1], lane)
                        self.set(x, lane.bottomRight[1], lane)
                    for y in range(lane.topLeft[1], lane.bottomRight[1]+1):
                        self.set(lane.topLeft[0], y, lane)
                        self.set(lane.bottomRight[0], y, lane)
        
        if len(sidewalks) != 0:
            for sidewalk in sidewalks:
                # for x in range(sidewalk.topLeft[0], sidewalk.bottomRight[0]+1):
                #     for y in range(sidewalk.topLeft[1], sidewalk.bottomRight[1]+1):
                #         self.set(x, y, sidewalk)
                for x in range(sidewalk.topLeft[0], sidewalk.bottomRight[0]+1):
                    self.set(x, sidewalk.topLeft[1], sidewalk)
                    self.set(x, sidewalk.bottomRight[1], sidewalk)
                for y in range(sidewalk.topLeft[1], sidewalk.bottomRight[1]+1):
                    self.set(sidewalk.topLeft[0], y, sidewalk)
                    self.set(sidewalk.bottomRight[0], y, sidewalk)
        
        if len(crosswalks) != 0:
            for crosswalk in crosswalks:
                # for x in range(crosswalk.topLeft[0], crosswalk.bottomRight[0]+1):
                #     for y in range(crosswalk.topLeft[1], crosswalk.bottomRight[1]+1):
                #         self.set(x, y, crosswalk)
                for x in range(crosswalk.topLeft[0], crosswalk.bottomRight[0]+1):
                    self.set(x, crosswalk.topLeft[1], crosswalk)
                    self.set(x, crosswalk.bottomRight[1], crosswalk)
                for y in range(crosswalk.topLeft[1], crosswalk.bottomRight[1]+1):
                    self.set(crosswalk.topLeft[0], y, crosswalk)
                    self.set(crosswalk.bottomRight[0], y, crosswalk)

        if len(vehicleAgents) != 0:
            for vehicle in vehicleAgents:
                for x in range(vehicle.topLeft[0], vehicle.bottomRight[0]+1):
                    for y in range(vehicle.topLeft[1], vehicle.bottomRight[1]+1):
                        self.set(x, y, vehicle)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agentIndex = -1
                agent_here = False
                for index in range(0, len(pedAgents)):
                    agent_here = np.array_equal(pedAgents[index].position, (i, j))
                    if agent_here:
                        agentIndex = index
                        break
                
                # agent_here = np.array_equal(agent_pos, (i, j))
                # agent_here = True

                # tile_img = PedGrid.render_tile(
                tile_img = Grid.render_tile(
                    cell,
                    # position=(i, j),
                    agent_dir=pedAgents[agentIndex].direction if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    # @classmethod
    # def render_tile(
    #     cls,
    #     obj,
    #     position: Tuple[int, int],
    #     agent_dir=None,
    #     highlight=False,
    #     tile_size=TILE_PIXELS,
    #     subdivs=3
    # ):
    #     """
    #     Render a tile and cache the result
    #     """

    #     # Hash map lookup key for the cache
    #     key = (agent_dir, highlight, tile_size)
    #     key = obj.encode() + key if obj else key

    #     if key in cls.tile_cache:
    #         return cls.tile_cache[key]

    #     img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    #     # Draw the grid lines (top and left edges)
    #     fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    #     fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    #     if obj != None:
    #         obj.render(img, position)

    #     # Overlay the agent on top
    #     if agent_dir is not None:
    #         tri_fn = point_in_triangle(
    #             (0.12, 0.19),
    #             (0.87, 0.50),
    #             (0.12, 0.81),
    #         )

    #         # Rotate the agent based on its direction
    #         tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
    #         fill_coords(img, tri_fn, (255, 0, 0))

    #     # Highlight the cell if needed
    #     if highlight:
    #         highlight_img(img)

    #     # Downsample the image to perform supersampling/anti-aliasing
    #     img = downsample(img, subdivs)

    #     # Cache the rendered tile
    #     cls.tile_cache[key] = img

    #     return img