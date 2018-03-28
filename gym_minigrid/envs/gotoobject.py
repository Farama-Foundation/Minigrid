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
        self.reward_range = (0, 1)

    def _genGrid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wallRect(0, 0, width, height)

        # Types and colors of objects we can generate
        types = ['key', 'ball', 'box']

        objs = []
        objPos = []

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
            objType = self._randElem(types)
            objColor = self._randElem(COLOR_NAMES)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)
            elif objType == 'box':
                obj = Box(objColor)

            pos = self.placeObj(obj)
            objs.append((objType, objColor))
            objPos.append(pos)

        # Randomize the player start position and orientation
        self.placeAgent()

        # Choose a random object to be picked up
        objIdx = self._randInt(0, len(objs))
        self.targetType, self.targetColor = objs[objIdx]
        self.targetPos = objPos[objIdx]

        descStr = '%s %s' % (self.targetColor, self.targetType)
        self.mission = 'go to the %s' % descStr
        #print(self.mission)

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        ax, ay = self.agentPos
        tx, ty = self.targetPos

        # Toggle/pickup action terminates the episode
        if action == self.actions.toggle:
            done = True

        # Reward performing the wait action next to the target object
        if action == self.actions.wait:
            if abs(ax - tx) <= 1 and abs(ay - ty) <= 1:
                reward = 1
            done = True

        return obs, reward, done, info

class GotoEnv8x8N2(GoToObjectEnv):
    def __init__(self):
        super().__init__(size=8, numObjs=2)

register(
    id='MiniGrid-GoToObject-6x6-N2-v0',
    entry_point='gym_minigrid.envs:GoToObjectEnv'
)

register(
    id='MiniGrid-GoToObject-8x8-N2-v0',
    entry_point='gym_minigrid.envs:GotoEnv8x8N2'
)
