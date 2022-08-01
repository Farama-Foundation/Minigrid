from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv
from gym_minigrid.register import register


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


register(
    id="MiniGrid-Empty-5x5-v0", entry_point="gym_minigrid.envs.empty:EmptyEnv", size=5
)

register(
    id="MiniGrid-Empty-Random-5x5-v0",
    entry_point="gym_minigrid.envs.empty:EmptyEnv",
    size=5,
    agent_start_pos=None,
)

register(
    id="MiniGrid-Empty-6x6-v0",
    entry_point="gym_minigrid.envs.empty:EmptyEnv",
    size=6,
)

register(
    id="MiniGrid-Empty-Random-6x6-v0",
    entry_point="gym_minigrid.envs.empty:EmptyEnv",
    size=6,
    agent_start_pos=None,
)

register(
    id="MiniGrid-Empty-8x8-v0",
    entry_point="gym_minigrid.envs.empty:EmptyEnv",
)

register(
    id="MiniGrid-Empty-16x16-v0",
    entry_point="gym_minigrid.envs.empty:EmptyEnv",
    size=16,
)
