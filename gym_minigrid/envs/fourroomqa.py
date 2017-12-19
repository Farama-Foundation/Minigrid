from gym_minigrid.minigrid import *
from gym_minigrid.register import register

"""
class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos
"""

class FourRoomQAEnv(MiniGridEnv):
    """
    Environment to experiment with embodied question answering
    https://arxiv.org/abs/1711.11543
    """

    # TODO: define actions




    def __init__(self, size=16):
        assert size >= 8
        super(FourRoomQAEnv, self).__init__(gridSize=size, maxSteps=8*size)

        # TODO: self.actions

        # TODO: self.action_space




    def _genGrid(self, width, height):
        grid = super(FourRoomQAEnv, self)._genGrid(width, height)

        # TODO: work in progress

        """
        # Create a vertical splitting wall
        splitIdx = self._randInt(2, gridSz-3)
        for i in range(0, gridSz):
            grid.set(splitIdx, i, Wall())

        # Place a door in the wall
        doorIdx = self._randInt(1, gridSz-2)
        grid.set(splitIdx, doorIdx, Door('yellow'))

        # Place a key on the left side
        #keyIdx = self._randInt(1 + gridSz // 2, gridSz-2)
        keyIdx = gridSz-2
        grid.set(1, keyIdx, Key('yellow'))
        """

        return grid





register(
    id='MiniGrid-FourRoomQA-v0',
    entry_point='gym_minigrid.envs:FourRoomQAEnv'
)
