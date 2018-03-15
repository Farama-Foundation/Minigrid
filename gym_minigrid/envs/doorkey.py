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
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wallRect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Create a vertical splitting wall
        splitIdx = self._randInt(2, width-2)
        self.grid.vertWall(splitIdx, 0)

        # Place the agent at a random position and orientation
        self.startPos = self._randPos(
            1, splitIdx,
            1, height-1
        )
        self.startDir = self._randInt(0, 4)

        # Place a door in the wall
        doorIdx = self._randInt(1, width-2)
        self.grid.set(splitIdx, doorIdx, LockedDoor('yellow'))

        # Place a yellow key on the left side
        while True:
            pos = self._randPos(
                1, splitIdx,
                1, height-1
            )
            if pos == self.startPos:
                continue
            if self.grid.get(*pos) != None:
                continue
            self.grid.set(*pos, Key('yellow'))
            break

        self.mission = "use the key to open the door and then get to the goal"

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
