from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class PutNearEnv(MiniGridEnv):
    """
    Environment in which the agent is instructed to place an object near
    another object through a natural language string.
    """

    def __init__(
        self,
        size=6,
        numObjs=2
    ):
        self.numObjs = numObjs
        super().__init__(gridSize=size, maxSteps=5*size)

        self.reward_range = (-1, 1)

    def _genGrid(self, width, height):
        # Create a grid surrounded by walls
        grid = Grid(width, height)
        for i in range(0, width):
            grid.set(i, 0, Wall())
            grid.set(i, height-1, Wall())
        for j in range(0, height):
            grid.set(0, j, Wall())
            grid.set(width-1, j, Wall())

        # Types and colors of objects we can generate
        types = ['key', 'ball']
        colors = list(COLORS.keys())

        objs = []
        objPos = []

        def nearObj(p1):
            for p2 in objPos:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if abs(dx) <= 1 and abs(dy) <= 1:
                    return True
            return False

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
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
                    self._randInt(1, width - 1),
                    self._randInt(1, height - 1)
                )
                if nearObj(pos):
                    continue
                if pos == self.startPos:
                    continue
                grid.set(*pos, obj)
                break

            objs.append((objType, objColor))
            objPos.append(pos)

        # Choose a random object to be moved up
        objIdx = self._randInt(0, len(objs))
        self.moveType, self.moveColor = objs[objIdx]
        self.movePos = objPos[objIdx]





        self.mission = 'put the %s %s near the Y' % (
            self.moveColor,
            self.moveType
        )

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

        """
        ax, ay = self.agentPos
        tx, ty = self.targetPos

        # Toggle/pickup action terminates the episode
        if action == self.actions.toggle:
            done = True

        # Reward performing the wait action next to the target object
        if action == self.actions.wait:
            if abs(ax - tx) <= 1 and abs(ay - ty) <= 1:
                reward = 1
            done = self.waitEnds
        """

        obs = self._observation(obs)

        return obs, reward, done, info

register(
    id='MiniGrid-PutNear-6x6-N2-v0',
    entry_point='gym_minigrid.envs:PutNearEnv'
)
