from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class GoToObjectEnv(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        size=6,
        numObjs=2
    ):
        self.numObjs = numObjs
        super().__init__(gridSize=size, maxSteps=5*size)
        self.reward_range = (-1000, 1000)

    def _genGrid(self, width, height):
        assert width == height
        gridSz = width

        # Create a grid surrounded by walls
        grid = Grid(width, height)
        for i in range(0, width):
            grid.set(i, 0, Wall())
            grid.set(i, height-1, Wall())
        for j in range(0, height):
            grid.set(0, j, Wall())
            grid.set(width-1, j, Wall())

        # Types and colors of objects we can generate
        types = ['key', 'ball', 'box']
        colors = list(COLORS.keys())

        objs = []

        # For each object to be generated
        for i in range(0, self.numObjs):
            objType = self._randElem(types)
            objColor = self._randElem(colors)

            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)
            elif objType == 'box':
                obj = Box(objColor)

            while True:
                pos = (
                    self._randInt(1, gridSz - 1),
                    self._randInt(1, gridSz - 1)
                )

                if pos != self.startPos:
                    grid.set(*pos, obj)
                    break

            objs.append(obj)

        # Choose a random object to be picked up
        target = objs[self._randInt(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = '%s %s' % (self.targetColor, self.targetType)

        """
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
        """

        self.mission = 'go to the %s' % descStr

        return grid

    def _observation(self, obs):
        """
        Encode observations
        """

        obs = {
            'image': obs,
            'mission': self.mission,
            'advice' : ''
        }

        return obs

    def _reset(self):
        obs = MiniGridEnv._reset(self)
        return self._observation(obs)

    def _step(self, action):
        obs, reward, done, info = MiniGridEnv._step(self, action)

        #if self.carrying:
        #    if self.carrying.color == self.targetColor and \
        #       self.carrying.type == self.targetType:
        #        reward = 1000 - self.stepCount
        #        done = True
        #    else:
        #        reward = -1000
        #        done = True

        obs = self._observation(obs)

        return obs, reward, done, info

register(
    id='MiniGrid-GoToObject-6x6-N2-v0',
    entry_point='gym_minigrid.envs:GoToObjectEnv'
)
