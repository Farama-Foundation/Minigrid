from gym import spaces
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Room:
    def __init__(self,
        top,
        size,
        doorPos
    ):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def randPos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._randPos(
            topX + 1, topX + sizeX - 1,
            topY + 1, topY + sizeY - 1
        )

class LockedRoom(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self
    ):
        size = 19
        super().__init__(gridSize=size, maxSteps=10*size)

        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        self.reward_range = (-1, 1)

    def _genGrid(self, width, height):
        # Create the grid
        grid = Grid(width, height)

        # Generate the surrounding walls
        for i in range(0, width):
            grid.set(i, 0, Wall())
            grid.set(i, height-1, Wall())
        for j in range(0, height):
            grid.set(0, j, Wall())
            grid.set(width-1, j, Wall())

        # Hallway walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        for j in range(0, height):
            grid.set(lWallIdx, j, Wall())
            grid.set(rWallIdx, j, Wall())

        self.rooms = []

        # Room splitting walls
        for n in range(0, 3):
            j = n * (height // 3)
            for i in range(0, lWallIdx):
                grid.set(i, j, Wall())
            for i in range(rWallIdx, width):
                grid.set(i, j, Wall())

            roomW = lWallIdx + 1
            roomH = height // 3 + 1
            self.rooms.append(Room(
                (0, j),
                (roomW, roomH),
                (lWallIdx, j + 3)
            ))
            self.rooms.append(Room(
                (rWallIdx, j),
                (roomW, roomH),
                (rWallIdx, j + 3)
            ))

        # Choose one random room to be locked
        lockedRoom = self._randElem(self.rooms)
        lockedRoom.locked = True
        goalPos = lockedRoom.randPos(self)
        grid.set(*goalPos, Goal())

        # Assign the door colors
        colors = set(COLOR_NAMES)
        for room in self.rooms:
            color = self._randElem(colors)
            colors.remove(color)
            room.color = color
            if room.locked:
                grid.set(*room.doorPos, LockedDoor(color))
            else:
                grid.set(*room.doorPos, Door(color))

        # Select a random room to contain the key
        while True:
            keyRoom = self._randElem(self.rooms)
            if keyRoom != lockedRoom:
                break
        keyPos = keyRoom.randPos(self)
        grid.set(*keyPos, Key(lockedRoom.color))

        # Randomize the player start position and orientation
        self.startPos = self._randPos(
            lWallIdx + 1, rWallIdx,
            1, height-1
        )
        self.startDir = self._randInt(0, 4)

        # Generate the mission string
        self.mission = (
            'get the %s key from the %s room, '
            'then use it to unlock the %s door '
            'so you can get to the goal'
        ) % (lockedRoom.color, keyRoom.color, lockedRoom.color)

        return grid

    def _observation(self, obs):
        """
        Encode observations
        """

        obs = {
            'image': obs,
            'mission': self.mission
        }

        return obs

    def _reset(self):
        obs = MiniGridEnv._reset(self)
        return self._observation(obs)

    def _step(self, action):
        obs, reward, done, info = MiniGridEnv._step(self, action)
        obs = self._observation(obs)
        return obs, reward, done, info

register(
    id='MiniGrid-LockedRoom-v0',
    entry_point='gym_minigrid.envs:LockedRoom'
)
