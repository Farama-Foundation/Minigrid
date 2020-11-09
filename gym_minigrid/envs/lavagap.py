from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class LavaGapEnv(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, size, obstacle_type=Lava, seed=None):
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=None
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # Generate and store random gap position
        self.gap_pos = np.array((
            self._rand_int(2, width - 2),
            self._rand_int(1, height - 1),
        ))

        # Place the obstacle wall
        self.grid.vert_wall(self.gap_pos[0], 1, height - 2, self.obstacle_type)

        # Put a hole in the wall
        self.grid.set(*self.gap_pos, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

class LavaGapS5Env(LavaGapEnv):
    def __init__(self):
        super().__init__(size=5)

class LavaGapS6Env(LavaGapEnv):
    def __init__(self):
        super().__init__(size=6)

class LavaGapS7Env(LavaGapEnv):
    def __init__(self):
        super().__init__(size=7)

register(
    id='MiniGrid-LavaGapS5-v0',
    entry_point='gym_minigrid.envs:LavaGapS5Env'
)

register(
    id='MiniGrid-LavaGapS6-v0',
    entry_point='gym_minigrid.envs:LavaGapS6Env'
)

register(
    id='MiniGrid-LavaGapS7-v0',
    entry_point='gym_minigrid.envs:LavaGapS7Env'
)
