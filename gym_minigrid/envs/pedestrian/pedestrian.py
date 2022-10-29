from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class PedestrianEnv(MiniGridEnv):
    def __init__(
        self,
        width=8,
        height=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            width=width,
            height=height,
            max_steps=4*width*height,
            # Set this to True for maximum speed
            see_through_walls=True
        )
    pass

    def _gen_grid(self, width, height):
        print("running gen grid method")
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        for i in range(1, height-1):
            self.put_obj(Goal(), width - 2, i)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        obs = super().gen_obs()
        obs['position'] = self.agent_pos
        return obs

class PedestrianEnv9x16(PedestrianEnv):
    def __init__(self):
        width = 16
        height = 9
        super().__init__(
            width=width,
            height=height,
            agent_start_pos=(1, height // 2),
            agent_start_dir=0
        )

class PedestrianEnv6x20(PedestrianEnv):
    def __init__(self):
        width = 60
        height = 200
        super().__init__(
            width=width,
            height=height,
            agent_start_pos=(1, height // 2),
            agent_start_dir=0
        )

class PedestrianEnv20x80(PedestrianEnv):
    def __init__(self):
        width = 20
        height = 80
        super().__init__(
            width=width,
            height=height,
            agent_start_pos=(1, height // 4),
            agent_start_dir=0
        )

        
register(
    id='Pedestrian-Empty-9x16-v0',
    entry_point='gym_minigrid.envs.pedestrian:PedestrianEnv9x16'
)
        
register(
    id='Pedestrian-Empty-6x20-v0',
    entry_point='gym_minigrid.envs.pedestrian:PedestrianEnv6x20'
)

register(
    id='Pedestrian-Empty-20x80-v0',
    entry_point='gym_minigrid.envs.pedestrian:PedestrianEnv20x80'
)