from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class DynamicObstaclesEnv(MiniGridEnv):
    """
    Empty grid environment with moving obstacles
    """

    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
            self.start_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        dirs = [-1, 0, 1]
        for obst in range(len(self.obstacles)):
            delta_x = random.choice(dirs)
            delta_y = random.choice(dirs)
            delta_pos = (delta_x, delta_y)
            new_pos = tuple(map(add, old_pos, delta_pos))

            while self.grid.get()

class DynamicObstaclesEnv5x5(MiniGridEnv):
    def __init__(self):
        super().__init__(size=5)


class DynamicObstaclesRandomEnv5x5(MiniGridEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)


class DynamicObstaclesEnv6x6(MiniGridEnv):
    def __init__(self):
        super().__init__(size=6)


class DynamicObstaclesRandomEnv6x6(MiniGridEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)


class DynamicObstaclesEnv16x16(MiniGridEnv):
    def __init__(self):
        super().__init__(size=16)


register(
    id='MiniGrid-Dynamic-Obstacles-5x5-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv5x5'
)

register(
    id='MiniGrid-Dynamic-Obstacles-Random-5x5-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesRandomEnv5x5'
)

register(
    id='MiniGrid-Dynamic-Obstacles-6x6-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv6x6'
)

register(
    id='MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesRandomEnv6x6'
)

register(
    id='MiniGrid-Dynamic-Obstacles-8x8-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv'
)

register(
    id='MiniGrid-Dynamic-Obstacles-16x16-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv16x16'
)