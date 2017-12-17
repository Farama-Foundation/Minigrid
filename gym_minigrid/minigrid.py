import math
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_minigrid.rendering import *

# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 32

# Number of cells (width and height) in the agent view
AGENT_VIEW_SIZE = 7

# Size of the array given as an observation to the agent
OBS_ARRAY_SIZE = (AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 3)

COLORS = {
    'red'   : (255, 0, 0),
    'green' : (0, 255, 0),
    'blue'  : (0, 0, 255),
    'purple': (112, 39, 195),
    'yellow': (255, 255, 0),
    'grey'  : (100, 100, 100)
}

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'empty'         : 0,
    'wall'          : 1,
    'door'          : 2,
    'locked_door'   : 3,
    'ball'          : 4,
    'key'           : 5,
    'goal'          : 6
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

    def canOverlap(self):
        """Can the agent overlap with this?"""
        return False

    def canPickup(self):
        """Can the agent pick this up?"""
        return False

    def canContain(self):
        """Can this contain another object?"""
        return False

    def toggle(self, env):
        """Method to trigger/toggle an action this object performs"""
        return False

    def render(self, r):
        assert False

    def _setColor(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

class Goal(WorldObj):
    def __init__(self):
        super(Goal, self).__init__('goal', 'green')

    def render(self, r):
        self._setColor(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super(Wall, self).__init__('wall', color)

    def render(self, r):
        self._setColor(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Door(WorldObj):
    def __init__(self, color, isOpen=False):
        super(Door, self).__init__('door', color)
        self.isOpen = isOpen

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(0, 0, 0)

        if self.isOpen:
            r.drawPolygon([
                (CELL_PIXELS-2, CELL_PIXELS),
                (CELL_PIXELS  , CELL_PIXELS),
                (CELL_PIXELS  ,           0),
                (CELL_PIXELS-2,           0)
            ])
            return

        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])
        r.drawPolygon([
            (2          , CELL_PIXELS-2),
            (CELL_PIXELS-2, CELL_PIXELS-2),
            (CELL_PIXELS-2,           2),
            (2          ,           2)
        ])
        r.drawCircle(CELL_PIXELS * 0.75, CELL_PIXELS * 0.5, 2)

    def toggle(self, env):
        if not self.isOpen:
            self.isOpen = True
            return True
        return False

    def canOverlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.isOpen

class LockedDoor(WorldObj):
    def __init__(self, color, isOpen=False):
        super(LockedDoor, self).__init__('locked_door', color)
        self.isOpen = isOpen

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(0, 0, 0)

        if self.isOpen:
            r.drawPolygon([
                (CELL_PIXELS-2, CELL_PIXELS),
                (CELL_PIXELS  , CELL_PIXELS),
                (CELL_PIXELS  ,           0),
                (CELL_PIXELS-2,           0)
            ])
            return

        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])
        r.drawPolygon([
            (2          , CELL_PIXELS-2),
            (CELL_PIXELS-2, CELL_PIXELS-2),
            (CELL_PIXELS-2,           2),
            (2          ,           2)
        ])
        r.drawLine(
            CELL_PIXELS * 0.75,
            CELL_PIXELS * 0.45,
            CELL_PIXELS * 0.75,
            CELL_PIXELS * 0.60
        )

    def toggle(self, env):
        # If the player has the right key to open the door
        if isinstance(env.carrying, Key) and env.carrying.color == self.color:
            self.isOpen = True
            # The key has been used, remove it from the agent
            env.carrying = None
            return True
        return False

    def canOverlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.isOpen

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def canPickup(self):
        return True

    def render(self, r):
        self._setColor(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)

class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def canPickup(self):
        return True

    def render(self, r):
        self._setColor(r)

        # Vertical quad
        r.drawPolygon([
            (16, 10),
            (20, 10),
            (20, 28),
            (16, 28)
        ])

        # Teeth
        r.drawPolygon([
            (12, 19),
            (16, 19),
            (16, 21),
            (12, 21)
        ])
        r.drawPolygon([
            (12, 26),
            (16, 26),
            (16, 28),
            (12, 28)
        ])

        r.drawCircle(18, 9, 6)
        r.setLineColor(0, 0, 0)
        r.setColor(0, 0, 0)
        r.drawCircle(18, 9, 2)

class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 4
        assert height >= 4

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def rotateLeft(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.width, self.height)

        for j in range(0, self.height):
            for i in range(0, self.width):
                v = self.get(self.width - 1 - j, i)
                grid.set(i, j, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    def render(self, r, tileSize):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tileSize: tile size in pixels
        """

        assert r.width == self.width * tileSize
        assert r.height == self.height * tileSize

        # Total grid size at native scale
        widthPx = self.width * CELL_PIXELS
        heightPx = self.height * CELL_PIXELS

        # Draw background (out-of-world) tiles the same colors as walls
        # so the agent understands these areas are not reachable
        c = COLORS['grey']
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])
        r.drawPolygon([
            (0    , heightPx),
            (widthPx, heightPx),
            (widthPx,      0),
            (0    ,      0)
        ])

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tileSize / CELL_PIXELS, tileSize / CELL_PIXELS)

        # Draw the background of the in-world cells black
        r.fillRect(
            0,
            0,
            widthPx,
            heightPx,
            0, 0, 0
        )

        # Draw grid lines
        r.setLineColor(100, 100, 100)
        for rowIdx in range(0, self.height):
            y = CELL_PIXELS * rowIdx
            r.drawLine(0, y, widthPx, y)
        for colIdx in range(0, self.width):
            x = CELL_PIXELS * colIdx
            r.drawLine(x, 0, x, heightPx)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                if cell == None:
                    continue
                r.push()
                r.translate(i * CELL_PIXELS, j * CELL_PIXELS)
                cell.render(r)
                r.pop()

        r.pop()

    def encode(self):
        """
        Produce a compact numpy encoding of the grid
        """

        codeSize = self.width * self.height * 3

        array = np.zeros(shape=(self.width, self.height, 3), dtype='uint8')

        for j in range(0, self.height):
            for i in range(0, self.width):

                v = self.get(i, j)

                if v == None:
                    continue

                array[i, j, 0] = OBJECT_TO_IDX[v.type]
                array[i, j, 1] = COLOR_TO_IDX[v.color]

                if hasattr(v, 'isOpen') and v.isOpen:
                    array[i, j, 2] = 1

        return array

    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width = array.shape[0]
        height = array.shape[1]
        assert array.shape[2] == 3

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):

                typeIdx  = array[i, j, 0]
                colorIdx = array[i, j, 1]
                openIdx  = array[i, j, 2]

                if typeIdx == 0:
                    continue

                objType = IDX_TO_OBJECT[typeIdx]
                color = IDX_TO_COLOR[colorIdx]
                isOpen = True if openIdx == 1 else 0

                if objType == 'wall':
                    v = Wall()
                elif objType == 'ball':
                    v = Ball(color)
                elif objType == 'key':
                    v = Key(color)
                elif objType == 'door':
                    v = Door(color, isOpen)
                elif objType == 'locked_door':
                    v = LockedDoor(color, isOpen)
                elif objType == 'goal':
                    v = Goal()
                else:
                    assert False, "unknown obj type in decode '%s'" % objType

                grid.set(i, j, v)

        return grid

class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second' : 10
    }

    # Possible actions
    NUM_ACTIONS = 4
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_FORWARD = 2
    ACTION_TOGGLE = 3

    def __init__(self, gridSize=16, maxSteps=100):
        # Renderer object used to render the whole grid (full-scale)
        self.gridRender = None

        # Renderer used to render observations (small-scale agent view)
        self.obsRender = None

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(MiniGridEnv.NUM_ACTIONS)

        # The observations are RGB images
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=OBS_ARRAY_SIZE
        )

        self.reward_range = (-1, 1000)

        # Environment configuration
        self.gridSize = gridSize
        self.maxSteps = maxSteps
        self.startPos = (1, 1)
        self.startDir = 0

        # Initialize the state
        self.seed()
        self.reset()

    def _genGrid(self, width, height):
        """
        Generate a new grid
        """

        # Initialize the grid
        grid = Grid(width, height)

        # Place walls around the edges
        for i in range(0, width):
            grid.set(i, 0, Wall())
            grid.set(i, height - 1, Wall())
        for j in range(0, height):
            grid.set(0, j, Wall())
            grid.set(height - 1, j, Wall())

        # Place a goal in the bottom-left corner
        grid.set(width - 2, height - 2, Goal())

        return grid

    def _reset(self):
        # Generate a new random grid at the start of each episode
        # To prevent this behavior, call env.seed with the same
        # seed before env.reset
        self.grid = self._genGrid(self.gridSize, self.gridSize)

        # Place the agent in the starting position and direction
        self.agentPos = self.startPos
        self.agentDir = self.startDir

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.stepCount = 0

        # Return first observation
        obs = self._genObs()
        return obs

    def _seed(self, seed=1337):
        """
        The seed function sets the random elements of the environment,
        and initializes the world.
        """

        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)

        return [seed]

    def _randInt(self, low, high):
        return self.np_random.randint(low, high)

    def _randElem(self, iterable):
        lst = list(iterable)
        idx = self._randInt(0, len(lst))
        return lst[idx]

    def getStepsRemaining(self):
        return self.maxSteps - self.stepCount

    def getDirVec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        # Pointing right
        if self.agentDir == 0:
            return (1, 0)
        # Down (positive Y)
        elif self.agentDir == 1:
            return (0, 1)
        # Pointing left
        elif self.agentDir == 2:
            return (-1, 0)
        # Up (negative Y)
        elif self.agentDir == 3:
            return (0, -1)
        else:
            assert False

    def getViewExts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.agentDir == 0:
            topX = self.agentPos[0]
            topY = self.agentPos[1] - AGENT_VIEW_SIZE // 2
        # Facing down
        elif self.agentDir == 1:
            topX = self.agentPos[0] - AGENT_VIEW_SIZE // 2
            topY = self.agentPos[1]
        # Facing right
        elif self.agentDir == 2:
            topX = self.agentPos[0] - AGENT_VIEW_SIZE + 1
            topY = self.agentPos[1] - AGENT_VIEW_SIZE // 2
        # Facing up
        elif self.agentDir == 3:
            topX = self.agentPos[0] - AGENT_VIEW_SIZE // 2
            topY = self.agentPos[1] - AGENT_VIEW_SIZE + 1
        else:
            assert False

        botX = topX + AGENT_VIEW_SIZE
        botY = topY + AGENT_VIEW_SIZE

        return (topX, topY, botX, botY)

    def _step(self, action):
        self.stepCount += 1

        reward = 0
        done = False

        # Rotate left
        if action == MiniGridEnv.ACTION_LEFT:
            self.agentDir -= 1
            if self.agentDir < 0:
                self.agentDir += 4

        # Rotate right
        elif action == MiniGridEnv.ACTION_RIGHT:
            self.agentDir = (self.agentDir + 1) % 4

        # Move forward
        elif action == MiniGridEnv.ACTION_FORWARD:
            u, v = self.getDirVec()
            newPos = (self.agentPos[0] + u, self.agentPos[1] + v)
            targetCell = self.grid.get(newPos[0], newPos[1])
            if targetCell == None or targetCell.canOverlap():
                self.agentPos = newPos
            elif targetCell.type == 'goal':
                done = True
                reward = 1000 - self.stepCount

        # Pick up or trigger/activate an item
        elif action == MiniGridEnv.ACTION_TOGGLE:
            u, v = self.getDirVec()
            cell = self.grid.get(self.agentPos[0] + u, self.agentPos[1] + v)
            if cell and cell.canPickup() and self.carrying is None:
                self.carrying = cell
                self.grid.set(self.agentPos[0] + u, self.agentPos[1] + v, None)
            elif cell:
                cell.toggle(self)

        else:
            assert False, "unknown action"

        if self.stepCount >= self.maxSteps:
            done = True

        obs = self._genObs()

        return obs, reward, done, {}

    def _genObs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        topX, topY, botX, botY = self.getViewExts()

        grid = self.grid.slice(topX, topY, AGENT_VIEW_SIZE, AGENT_VIEW_SIZE)

        for i in range(self.agentDir + 1):
            grid = grid.rotateLeft()

        obs = grid.encode()

        return obs

    def getObsRender(self, obs):
        """
        Render an agent observation for visualization
        """

        if self.obsRender == None:
            self.obsRender = Renderer(
                AGENT_VIEW_SIZE * CELL_PIXELS // 2,
                AGENT_VIEW_SIZE * CELL_PIXELS // 2
            )

        r = self.obsRender

        r.beginFrame()

        grid = Grid.decode(obs)

        # Render the whole grid
        grid.render(r, CELL_PIXELS // 2)

        # Draw the agent
        r.push()
        r.scale(0.5, 0.5)
        r.translate(
            CELL_PIXELS * (0.5 + AGENT_VIEW_SIZE // 2),
            CELL_PIXELS * (AGENT_VIEW_SIZE - 0.5)
        )
        r.rotate(3 * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            ( 12,  0),
            (-12, -10)
        ])
        r.pop()

        r.endFrame()

        return r.getPixmap()

    def _render(self, mode='human', close=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.gridRender:
                self.gridRender.close()
            return

        if self.gridRender is None:
            self.gridRender = Renderer(
                self.gridSize * CELL_PIXELS,
                self.gridSize * CELL_PIXELS,
                True if mode == 'human' else False
            )

        r = self.gridRender

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, CELL_PIXELS)

        # Draw the agent
        r.push()
        r.translate(
            CELL_PIXELS * (self.agentPos[0] + 0.5),
            CELL_PIXELS * (self.agentPos[1] + 0.5)
        )
        r.rotate(self.agentDir * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            ( 12,  0),
            (-12, -10)
        ])
        r.pop()

        # Highlight what the agent can see
        topX, topY, botX, botY = self.getViewExts()
        r.fillRect(
            topX * CELL_PIXELS,
            topY * CELL_PIXELS,
            AGENT_VIEW_SIZE * CELL_PIXELS,
            AGENT_VIEW_SIZE * CELL_PIXELS,
            200, 200, 200, 75
        )

        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()

        return r
