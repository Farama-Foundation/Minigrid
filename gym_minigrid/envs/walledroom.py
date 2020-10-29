from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class WalledEnv(MiniGridEnv):
    """
    Walled grid environment, one big obstacle, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=False
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place a big rectangular wall in the middle
        # self.grid.wall_rect(3, 3, width-2, height-2)
        # This big wall is kind of not good
        # self.grid.wall_rect(3, 3, width-6, height-6)
        # The vertical wall
        if self.grid_size == 16:
            self.grid.vert_wall(12, 3, length=height-6)
            self.grid.vert_wall(11, 3, length=height-8)
            self.grid.vert_wall(10, 3, length=height-6)
            self.grid.vert_wall(9, 4, length=height-8)
            self.grid.vert_wall(6, 5, length=height-8)
            self.grid.vert_wall(5, 5, length=height-8)
            self.grid.vert_wall(4, 5, length=height-8)
            self.grid.vert_wall(3, 3, length=height-8)
        elif self.grid_size == 32:
            self.grid.vert_wall(26, 6, length=height-12)
            self.grid.vert_wall(25, 9, length=height-14)
            self.grid.vert_wall(24, 8, length=height-13)
            self.grid.vert_wall(23, 9, length=height-14)
            self.grid.vert_wall(22, 8, length=height-13)
            self.grid.vert_wall(18, 7, length=height-11)
            self.grid.vert_wall(17, 7, length=height-12)
            self.grid.vert_wall(16, 8, length=height-11)
            self.grid.vert_wall(15, 9, length=height-12)
            self.grid.vert_wall(11, 3, length=height-10)
            self.grid.vert_wall(10, 3, length=height-10)
            self.grid.vert_wall(9, 4, length=height-10)
            self.grid.vert_wall(8, 6, length=height-10)
            self.grid.vert_wall(7, 3, length=height-10)

        self.mission = "get to the green goal square"


class WalledEnv6x6(WalledEnv):
    def __init__(self):
        super().__init__(size=6)


class WalledEnv16x16(WalledEnv):
    def __init__(self):
        super().__init__(size=16)


class WalledEnv32x32(WalledEnv):
    def __init__(self):
        super().__init__(size=32)


register(
    id='MiniGrid-Walled-6x6-v0',
    entry_point='gym_minigrid.envs:WalledEnv6x6'
)

register(
    id='MiniGrid-Walled-8x8-v0',
    entry_point='gym_minigrid.envs:WalledEnv'
)

register(
    id='MiniGrid-Walled-16x16-v0',
    entry_point='gym_minigrid.envs:WalledEnv16x16'
)

register(
    id='MiniGrid-Walled-32x32-v0',
    entry_point='gym_minigrid.envs:WalledEnv32x32'
)
