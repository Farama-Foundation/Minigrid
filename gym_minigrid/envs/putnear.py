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
        self.reward_range = (0, 1)

    def _genGrid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horzWall(0, 0)
        self.grid.horzWall(0, height-1)
        self.grid.vertWall(0, 0)
        self.grid.vertWall(width-1, 0)

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
                self.grid.set(*pos, obj)
                break

            objs.append((objType, objColor))
            objPos.append(pos)

        # Choose a random object to be moved
        objIdx = self._randInt(0, len(objs))
        self.moveType, self.moveColor = objs[objIdx]
        self.movePos = objPos[objIdx]

        # Choose a target object (to put the first object next to)
        while True:
            targetIdx = self._randInt(0, len(objs))
            if targetIdx != objIdx:
                break
        self.targetType, self.targetColor = objs[targetIdx]
        self.targetPos = objPos[targetIdx]

        self.mission = 'put the %s %s near the %s %s' % (
            self.moveColor,
            self.moveType,
            self.targetColor,
            self.targetType
        )

    def step(self, action):
        preCarrying = self.carrying

        obs, reward, done, info = MiniGridEnv.step(self, action)

        u, v = self.getDirVec()
        ox, oy = (self.agentPos[0] + u, self.agentPos[1] + v)
        tx, ty = self.targetPos

        # Pickup/drop action
        if action == self.actions.toggle:
            # If we picked up the wrong object, terminate the episode
            if self.carrying:
                if self.carrying.type != self.moveType or self.carrying.color != self.moveColor:
                    done = True

            # If successfully dropping an object near the target
            if preCarrying:
                if self.grid.get(ox, oy) is preCarrying:
                    if abs(ox - tx) <= 1 and abs(oy - ty) <= 1:
                        reward = 1
                done = True

        return obs, reward, done, info

class PutNear8x8N3(PutNearEnv):
    def __init__(self):
        super().__init__(size=8, numObjs=3)

register(
    id='MiniGrid-PutNear-6x6-N2-v0',
    entry_point='gym_minigrid.envs:PutNearEnv'
)

register(
    id='MiniGrid-PutNear-8x8-N3-v0',
    entry_point='gym_minigrid.envs:PutNear8x8N3'
)
