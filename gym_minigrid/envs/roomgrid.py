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

        # Set of rooms this is connected to
        self.neighbors = set()

        # List of objects contained
        self.objs = []

    def randPos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._randPos(
            topX + 1, topX + sizeX - 1,
            topY + 1, topY + sizeY - 1
        )

class RoomGrid(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This is meant to serve as a base class for other environments.
    """

    def __init__(
            self,
            roomSize=6,
            numCols=4,
            maxObsPerRoom=3,
            lockedRooms=False
    ):
        assert roomSize > 0
        assert roomSize >= 4
        assert numCols > 0
        self.roomSize = roomSize
        self.numCols = numCols
        self.numRows = numCols
        self.maxObsPerRoom =  maxObsPerRoom
        self.lockedRooms = False

        gridSize = (roomSize - 1) * numCols + 1
        super().__init__(gridSize=gridSize, maxSteps=6*gridSize)

        self.reward_range = (0, 1)

    def getRoom(self, x, y):
        """Get the room a given position maps to"""

        assert x >= 0
        assert y >= 0

        i = x // self.roomSize
        j = y // self.roomSize

        assert i < self.numCols
        assert j < self.numRows

        return self.roomGrid[j][i]

    def _genGrid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        self.roomGrid = []
        self.rooms = []

        # For each row of rooms
        for j in range(0, self.numRows):
            row = []

            # For each column of rooms
            for i in range(0, self.numCols):
                room = Room(
                    (i * (self.roomSize-1), j * (self.roomSize-1)),
                    (self.roomSize, self.roomSize)
                )

                row.append(room)
                self.rooms.append(room)

                # Generate the walls for this room
                self.grid.wallRect(*room.top, *room.size)

            self.roomGrid.append(row)

        # Randomize the player start position and orientation
        self.placeAgent()

        # Find which room the agent was placed in
        startRoom = self.getRoom(*self.startPos)







        # TODO: respect maxObsPerRoom

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
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

register(
    id='MiniGrid-RoomGrid-v0',
    entry_point='gym_minigrid.envs:RoomGrid'
)
