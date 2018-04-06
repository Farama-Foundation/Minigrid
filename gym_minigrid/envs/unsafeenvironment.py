from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class UnsafeEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(gridSize=size, maxSteps=3*size)

    def _genGrid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horzWall(0, 0)
        self.grid.horzWall(0, height-1)
        self.grid.vertWall(0, 0)
        self.grid.vertWall(width-1, 0)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())
        self.grid.set(width - 4, height - 4, Water())
        self.grid.set(width - 5, height - 4, Water())
        self.grid.set(width - 6, height - 4, Water())
        self.grid.set(width - 7, height - 4, Water())


        self.mission = "get to the green goal square without moving on water"

class UnsafeEnv6x6(UnsafeEnv):
    def __init__(self):
        super().__init__(size=6)

class UnsafeEnv16x16(UnsafeEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Unsafeevironment-6x6-v0',
    entry_point='gym_minigrid.envs:UnsafeEnv6x6'
)

register(
    id='MiniGrid-Unsafeevironment-8x8-v0',
    entry_point='gym_minigrid.envs:UnsafeEnv'
)

register(
    id='MiniGrid-Unsafeevironment-16x16-v0',
    entry_point='gym_minigrid.envs:UnsafeEnv16x16'
)
