from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(gridSize=size, maxSteps=3 * size)

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

class DoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(gridSize=size, maxSteps=4 * size)

    def _genGrid(self, width, height):
        grid = super()._genGrid(width, height)
        assert width == height
        gridSz = width

        # Create a vertical splitting wall
        splitIdx = self._randInt(2, gridSz-2)
        for i in range(0, gridSz):
            grid.set(splitIdx, i, Wall())

        # Place the agent at a random position and orientation
        self.startPos = (
            self._randInt(1, splitIdx),
            self._randInt(1, gridSz-1)
        )
        self.startDir = self._randInt(0, 4)

        # Place a door in the wall
        doorIdx = self._randInt(1, gridSz-2)
        grid.set(splitIdx, doorIdx, LockedDoor('yellow'))

        # Place a yellow key on the left side
        while True:
            pos = (
                self._randInt(1, splitIdx),
                self._randInt(1, gridSz-1)
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
