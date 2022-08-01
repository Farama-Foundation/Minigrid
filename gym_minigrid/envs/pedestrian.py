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

        
register(
    id='Pedestrian-Empty-9x16-v0',
    entry_point='gym_minigrid.envs:PedestrianEnv9x16'
)
