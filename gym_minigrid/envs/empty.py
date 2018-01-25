from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(gridSize=size, maxSteps=3*size)

    def _genGrid(self, width, height):
        # Create an empty grid
        grid = Grid(width, height)

        # Place walls around the edges
        for i in range(0, width):
            grid.set(i, 0, Wall())
            grid.set(i, height - 1, Wall())
        for j in range(0, height):
            grid.set(0, j, Wall())
            grid.set(height - 1, j, Wall())

        # Place a goal in the bottom-right corner
        grid.set(width - 2, height - 2, Goal())

        return grid

class EmptyEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Empty-6x6-v0',
    entry_point='gym_minigrid.envs:EmptyEnv6x6'
)

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)

register(
    id='MiniGrid-Empty-16x16-v0',
    entry_point='gym_minigrid.envs:EmptyEnv16x16'
)
