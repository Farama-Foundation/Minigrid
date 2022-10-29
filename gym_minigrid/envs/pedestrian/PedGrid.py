import numpy as np
from gym_minigrid.minigrid import Grid
from typing import List
from gym_minigrid.agents import PedAgent


class PedGrid(Grid):

    def render(
        self,
        tile_size,
        agents: List[PedAgent],
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

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agentIndex = -1
                agent_here = False
                for index in range(0, len(agents)):
                    agent_here = np.array_equal(agents[index].position, (i, j))
                    if agent_here:
                        agentIndex = index
                        break
                # agent_here = np.array_equal(agent_pos, (i, j))
                # agent_here = True
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agents[agentIndex].direction if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

        