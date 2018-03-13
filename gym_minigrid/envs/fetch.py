from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class FetchEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings
    """

    def __init__(
        self,
        size=8,
        numObjs=3
    ):
        self.numObjs = numObjs
        super().__init__(gridSize=size, maxSteps=5*size)
        self.reward_range = (0, 1)

    def _genGrid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horzWall(0, 0)
        self.grid.horzWall(0, height-1)
        self.grid.vertWall(0, 0)
        self.grid.vertWall(width-1, 0)

        types = ['key', 'ball']

        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objType = self._randElem(types)
            objColor = self._randElem(COLOR_NAMES)

            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)

            self.placeObj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.placeAgent()

        # Choose a random object to be picked up
        target = objs[self._randInt(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = '%s %s' % (self.targetColor, self.targetType)

        # Generate the mission string
        idx = self._randInt(0, 5)
        if idx == 0:
            self.mission = 'get a %s' % descStr
        elif idx == 1:
            self.mission = 'go get a %s' % descStr
        elif idx == 2:
            self.mission = 'fetch a %s' % descStr
        elif idx == 3:
            self.mission = 'go fetch a %s' % descStr
        elif idx == 4:
            self.mission = 'you must fetch a %s' % descStr
        assert hasattr(self, 'mission')

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.color == self.targetColor and \
               self.carrying.type == self.targetType:
                reward = 1
                done = True
            else:
                reward = 0
                done = True

        return obs, reward, done, info

class FetchEnv5x5N2(FetchEnv):
    def __init__(self):
        super().__init__(size=5, numObjs=2)

class FetchEnv6x6N2(FetchEnv):
    def __init__(self):
        super().__init__(size=6, numObjs=2)

register(
    id='MiniGrid-Fetch-5x5-N2-v0',
    entry_point='gym_minigrid.envs:FetchEnv5x5N2'
)

register(
    id='MiniGrid-Fetch-6x6-N2-v0',
    entry_point='gym_minigrid.envs:FetchEnv6x6N2'
)

register(
    id='MiniGrid-Fetch-8x8-N3-v0',
    entry_point='gym_minigrid.envs:FetchEnv'
)
