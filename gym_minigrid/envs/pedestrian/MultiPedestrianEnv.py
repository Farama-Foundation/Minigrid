from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.agents import PedAgent

class MultiPedestrianEnv(MiniGridEnv):
    def __init__(
        self,
        width=8,
        height=8
    ):
        self.peds = []
        # self.agent_start_pos = agent_start_pos
        # self.agent_start_dir = agent_start_dir

        super().__init__(
            width=width,
            height=height,
            max_steps=4*width*height,
            # Set this to True for maximum speed
            see_through_walls=True
        )
    pass

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

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
        obs['position'] = self.agent_pos ## TODO, Terry plan the observation structure
        return obs

    def addPedestrian(self, ped: PedAgent):
        self.peds.append(ped)


class MultiPedestrianEnv6x20(MultiPedestrianEnv):
    def __init__(self):
        width = 60
        height = 200
        super().__init__(
            width=width,
            height=height
        )

register(
    id='MultiPedestrian-Empty-9x16-v0',
    entry_point='gym_minigrid.envs:PedestrianEnv9x16'
)
