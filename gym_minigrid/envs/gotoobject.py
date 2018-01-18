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
        objPos = []

        # For each object to be generated
        for i in range(0, self.numObjs):
            objType = self._randElem(types)
            objColor = self._randElem(colors)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

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

            objs.append((objType, objColor))
            objPos.append(pos)

        # Choose a random object to be picked up
        objIdx = self._randInt(0, len(objs))
        self.targetType, self.targetColor = objs[objIdx]
        self.targetPos = objPos[objIdx]

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
        #print(self.mission)

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

        ax, ay = self.agentPos
        tx, ty = self.targetPos

        # Reward being next to the object
        # Double reward waiting next to the object
        if abs(ax - tx) <= 1 and abs(ay - ty) <= 1:
            if action == self.actions.wait:
                reward = 2
            else:
                reward = 1

        obs = self._observation(obs)

        return obs, reward, done, info

register(
    id='MiniGrid-GoToObject-6x6-N2-v0',
    entry_point='gym_minigrid.envs:GoToObjectEnv'
)
