from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Room:
    def __init__(self,
        top,
        size
    ):
        # Top-left corner and size (tuples)
        self.top = top
        self.size = size

        # List of door objects and door positions
        self.doors = []
        self.doorPos = []

        # Indicates if this room is locked
        self.locked = False

        # TODO: connectivity?

        # List of objects contained
        self.objs = []

    def randPos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._randPos(
            topX + 1, topX + sizeX - 1,
            topY + 1, topY + sizeY - 1
        )

class PlaygroundV1(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.
    """

    def __init__(
            self,
            roomSize=6,
            numCols=4
    ):
        assert roomSize > 0
        assert roomSize >= 4
        assert numCols > 0
        self.roomSize = roomSize
        self.numCols = numCols
        self.numRows = numCols

        gridSize = (roomSize - 1) * numCols + 1
        super().__init__(gridSize=gridSize, maxSteps=6*gridSize)
        self.reward_range = (0, 1)

    def _genGrid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horzWall(0, 0)
        self.grid.horzWall(0, height-1)
        self.grid.vertWall(0, 0)
        self.grid.vertWall(width-1, 0)

        roomW = self.roomSize
        roomH = self.roomSize

        self.rooms = []

        # Generate the list of rooms
        for j in range(0, self.numRows):
            for i in range(0, self.numCols):
                room = Room(
                    (i * (self.roomSize-1), j * (self.roomSize-1)),
                    (self.roomSize, self.roomSize)
                )
                self.rooms.append(room)

        # TODO: generate walls
        # May want to add function to Grid class, wallRect(i, j, w, h, color)









        # Randomize the player start position and orientation
        self.placeAgent()

        # Place random objects in the world
        types = ['key', 'ball', 'box']
        for i in range(0, 12):
            objType = self._randElem(types)
            objColor = self._randElem(COLOR_NAMES)
            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)
            elif objType == 'box':
                obj = Box(objColor)
            self.placeObj(obj)

        # TODO: curriculum generation
        self.mission = ''

    def step(self, action):
        obs, reward, done, info = super().step(self, action)
        return obs, reward, done, info

register(
    id='MiniGrid-Playground-v1',
    entry_point='gym_minigrid.envs:PlaygroundV1'
)
