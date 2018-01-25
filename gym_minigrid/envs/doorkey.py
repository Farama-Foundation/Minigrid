from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class DoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(gridSize=size, maxSteps=4 * size)

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

        # Create a vertical splitting wall
        splitIdx = self._randInt(2, width-2)
        for i in range(0, height):
            grid.set(splitIdx, i, Wall())

        # Place the agent at a random position and orientation
        self.startPos = self._randPos(
            1, splitIdx,
            1, height-1
        )
        self.startDir = self._randInt(0, 4)

        # Place a door in the wall
        doorIdx = self._randInt(1, width-2)
        grid.set(splitIdx, doorIdx, LockedDoor('yellow'))

        # Place a yellow key on the left side
        while True:
            pos = self._randPos(
                1, splitIdx,
                1, height-1
            )
            if pos == self.startPos:
                continue
            if grid.get(*pos) != None:
                continue
            grid.set(*pos, Key('yellow'))
            break

        return grid

class DoorKeyEnv5x5(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)

class DoorKeyEnv6x6(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)

class DoorKeyEnv16x16(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-DoorKey-5x5-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv5x5'
)

register(
    id='MiniGrid-DoorKey-6x6-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv6x6'
)

register(
    id='MiniGrid-DoorKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv'
)

register(
    id='MiniGrid-DoorKey-16x16-v0',
    entry_point='gym_minigrid.envs:DoorKeyEnv16x16'
)
