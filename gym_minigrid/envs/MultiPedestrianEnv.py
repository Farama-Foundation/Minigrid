from typing import List
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import PedAgent

class MultiPedestrianEnv(MiniGridEnv):
    def __init__(
        self,
        agents: List[PedAgent],
        width=8,
        height=8,
    ):
        # self.agent_start_pos = agent_start_pos
        # self.agent_start_dir = agent_start_dir
        self.agents = agents

        super().__init__(
            width=width,
            height=height,
            max_steps=4*width*height,
            # Set this to True for maximum speed
            see_through_walls=True
        )
    pass

    def _gen_grid(self, width, height):
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        # Create an empty grid
        self.grid = PedGrid(width, height)
        # self.grid = PedGrid(width, height, self.agents)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        for i in range(1, height-1):
            self.put_obj(Goal(), width - 2, i)

        # # Place the agent
        # if self.agent_start_pos is not None:
        #     self.agent_pos = self.agent_start_pos
        #     self.agent_dir = self.agent_start_dir
        # else:
        #     self.place_agent()

        self.mission = "switch sidewalks"

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        obs = super().gen_obs()
        obs['positions'] = self.agent_pos #TODO, Terry plan the observation structure
        return obs

    def addPedestrian(self, ped: PedAgent):
        self.agents.append(ped)

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # # Compute which cells are visible to the agent
        # _, vis_mask = self.gen_obs_grid()

        # # Compute the world coordinates of the bottom-left corner
        # # of the agent's view area
        # f_vec = self.dir_vec
        # r_vec = self.right_vec
        # top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # # Mask of which cells to highlight
        # highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # # For each cell in the visibility mask
        # for vis_j in range(0, self.agent_view_size):
        #     for vis_i in range(0, self.agent_view_size):
        #         # If this cell is not visible, don't highlight it
        #         if not vis_mask[vis_i, vis_j]:
        #             continue

        #         # Compute the world coordinates of this cell
        #         abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

        #         if abs_i < 0 or abs_i >= self.width:
        #             continue
        #         if abs_j < 0 or abs_j >= self.height:
        #             continue

        #         # Mark this cell to be highlighted
        #         highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agents,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=None
            # highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    # @property
    # def dir_vec(self, index):
    #     """
    #     Get the direction vector for the agent, pointing in the direction
    #     of forward movement.
    #     """

    #     assert self.agents[index].direction >= 0 and self.agents[index].direction < 4
    #     return DIR_TO_VEC[self.agents[index].direction]

    # @property
    # def front_pos(self, index):
    #     """
    #     Get the position of the cell that is right in front of the agent
    #     """

    #     return self.agents[index].position + self.dir_vec(index)

    def step(self, actions):
        self.step_count += 1

        reward = 0
        done = False

        for index in range(0, len(self.agents)):
            # Rotate left
            if actions[index] == self.actions.left:
                self.agents[index].direction -= 1
                if self.agents[index].direction < 0:
                    self.agents[index].direction += 4

            # Rotate right
            elif actions[index] == self.actions.right:
                self.agents[index].direction = (self.agents[index].direction + 1) % 4
                
            # Move forward
            elif actions[index] == self.actions.forward:
                # Get the position in front of the agent
                # fwd_pos = self.front_pos(index)
                fwd_pos = self.agents[index].position + DIR_TO_VEC[self.agents[index].direction]
                # Get the contents of the cell in front of the agent
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell == None or fwd_cell.can_overlap():
                    self.agents[index].position = fwd_pos

            else:
                assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def getNumAgents(self):
        return len(self.agents)

class PedGrid(Grid):

    # def __init__(
    #     self,
    #     width,
    #     height,
    #     agents: List[PedAgent]
    # ):
    #     agents = agents

    #     super().__init__(
    #         width=width,
    #         height=height
    #     )

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

class MultiPedestrianEnv20x80(MultiPedestrianEnv):
    def __init__(self):
        # width = 20
        # height = 80
        width = 10
        height = 20
        super().__init__(
            width=width,
            height=height,
            agents=[PedAgent((5, 5), 3, 2), PedAgent((7, 10), 3, 2), PedAgent((2, 5), 3, 2)]
        )

register(
    id='MultiPedestrian-Empty-20x80-v0',
    entry_point='gym_minigrid.envs.MultiPedestrianEnv:MultiPedestrianEnv20x80'
)
