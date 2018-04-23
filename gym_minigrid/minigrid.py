import math
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from gym_minigrid.rendering import *

# Size in pixels of a cell in the full-scale human view
CELL_PIXELS = 32

# Number of cells (width and height) in the agent view
AGENT_VIEW_SIZE = 7

# Size of the array given as an observation to the agent
OBS_ARRAY_SIZE = (AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, 3)

# Map of color names to RGB values
COLORS = {
    'red'   : (255, 0, 0),
    'green' : (0, 255, 0),
    'blue'  : (0, 0, 255),
    'purple': (112, 39, 195),
    'yellow': (255, 255, 0),
    'grey'  : (100, 100, 100)
}

COLOR_NAMES = sorted(list(COLORS.keys()))

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
    'key'           : 4,
    'ball'          : 5,
    'box'           : 6,
    'goal'          : 7
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

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

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def _set_color(self, r):
        """Set the color of this object as the active drawing color"""
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

class Goal(WorldObj):
    def __init__(self):
        super(Goal, self).__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super(Wall, self).__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Door(WorldObj):
    def __init__(self, color, is_open=False):
        super(Door, self).__init__('door', color)
        self.is_open = is_open

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        if not self.is_open:
            self.is_open = True
            return True
        return False

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(0, 0, 0)

        if self.is_open:
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

class LockedDoor(WorldObj):
    def __init__(self, color, is_open=False):
        super(LockedDoor, self).__init__('locked_door', color)
        self.is_open = is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if isinstance(env.carrying, Key) and env.carrying.color == self.color:
            self.is_open = True
            # The key has been used, remove it from the agent
            env.carrying = None
            return True
        return False

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2], 50)

        if self.is_open:
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
            CELL_PIXELS * 0.55,
            CELL_PIXELS * 0.5,
            CELL_PIXELS * 0.75,
            CELL_PIXELS * 0.5
        )

class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, r):
        self._set_color(r)

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

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)

class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(0, 0, 0)
        r.setLineWidth(2)

        r.drawPolygon([
            (4            , CELL_PIXELS-4),
            (CELL_PIXELS-4, CELL_PIXELS-4),
            (CELL_PIXELS-4,             4),
            (4            ,             4)
        ])

        r.drawLine(
            4,
            CELL_PIXELS / 2,
            CELL_PIXELS - 4,
            CELL_PIXELS / 2
        )

        r.setLineWidth(1)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

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

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

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

    def horz_wall(self, x, y, length=None):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, Wall())

    def vert_wall(self, x, y, length=None):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, Wall())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
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

    def render(self, r, tile_size):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        assert r.width == self.width * tile_size
        assert r.height == self.height * tile_size

        # Total grid size at native scale
        widthPx = self.width * CELL_PIXELS
        heightPx = self.height * CELL_PIXELS

        r.push()

        # Internally, we draw at the "large" full-grid resolution, but we
        # use the renderer to scale back to the desired size
        r.scale(tile_size / CELL_PIXELS, tile_size / CELL_PIXELS)

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

                if hasattr(v, 'is_open') and v.is_open:
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
                is_open = True if openIdx == 1 else 0

                if objType == 'wall':
                    v = Wall(color)
                elif objType == 'ball':
                    v = Ball(color)
                elif objType == 'key':
                    v = Key(color)
                elif objType == 'box':
                    v = Box(color)
                elif objType == 'door':
                    v = Door(color, is_open)
                elif objType == 'locked_door':
                    v = LockedDoor(color, is_open)
                elif objType == 'goal':
                    v = Goal()
                else:
                    assert False, "unknown obj type in decode '%s'" % objType

                grid.set(i, j, v)

        return grid

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(1, grid.height)):
            for i in range(0, grid.width-1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i+1, j] = True
                mask[i+1, j-1] = True
                mask[i, j-1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i-1, j-1] = True
                mask[i-1, j] = True
                mask[i, j-1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask

class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'pixmap'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

        # Wait/stay put/do nothing
        wait = 6

    def __init__(
        self,
        grid_size=16,
        max_steps=100,
        see_through_walls=False,
        seed=1337
    ):
        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=OBS_ARRAY_SIZE,
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None

        # Renderer used to render observations (small-scale agent view)
        self.obs_render = None

        # Environment configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Starting position and direction for the agent
        self.start_pos = None
        self.start_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.grid_size, self.grid_size)

        # These fields should be defined by _gen_grid
        assert self.start_pos is not None
        assert self.start_dir is not None

        # Check that the agent doesn't overlap with an object
        assert self.grid.get(*self.start_pos) is None

        # Place the agent in the starting position and direction
        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        The agent is represented by `⏩`. A grid pixel is represented by 2-character
        string, the first one for the object and the second one for the color.
        """

        from copy import deepcopy

        def rotate_left(array):
            new_array = deepcopy(array)
            for i in range(len(array)):
                for j in range(len(array[0])):
                    new_array[j][len(array[0])-1-i] = array[i][j]
            return new_array

        def vertically_symmetrize(array):
            new_array = deepcopy(array)
            for i in range(len(array)):
                for j in range(len(array[0])):
                    new_array[i][len(array[0])-1-j] = array[i][j]
            return new_array

        # Map of object id to short string
        OBJECT_IDX_TO_IDS = {
            0: ' ',
            1: 'W',
            2: 'D',
            3: 'L',
            4: 'K',
            5: 'B',
            6: 'X',
            7: 'G'
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map of color id to short string
        COLOR_IDX_TO_IDS = {
            0: 'R',
            1: 'G',
            2: 'B',
            3: 'P',
            4: 'Y',
            5: 'E'
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_IDS = {
            0: '⏩',
            1: '⏬',
            2: '⏪',
            3: '⏫'
        }

        array = self.grid.encode()

        array = rotate_left(array)
        array = vertically_symmetrize(array)

        new_array = []

        for line in array:
            new_line = []

            for pixel in line:
                # If the door is opened
                if pixel[0] in [2, 3] and pixel[2] == 1:
                    object_ids = OPENDED_DOOR_IDS
                else:
                    object_ids = OBJECT_IDX_TO_IDS[pixel[0]]

                # If no object
                if pixel[0] == 0:
                    color_ids = ' '
                else:
                    color_ids = COLOR_IDX_TO_IDS[pixel[1]]

                new_line.append(object_ids + color_ids)

            new_array.append(new_line)

        # Add the agent
        new_array[self.agent_pos[1]][self.agent_pos[0]] = AGENT_DIR_TO_IDS[self.agent_dir]

        return "\n".join([" ".join(line) for line in new_array])

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self, obj, top=None, size=None, reject_fn=None):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)

        if size is None:
            size = (self.grid.width, self.grid.height)

        while True:
            pos = np.array((
                self._rand_int(top[0], top[0] + size[0]),
                self._rand_int(top[1], top[1] + size[1])
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.start_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        return pos

    def place_agent(self, top=None, size=None, rand_dir=True):
        """
        Set the agent's starting point at an empty position in the grid
        """

        pos = self.place_obj(None, top, size)
        self.start_pos = pos

        if rand_dir:
            self.start_dir = self._rand_int(0, 4)

        return pos

    def get_dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    def get_right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.get_dir_vec()
        return np.array((-dy, dx))

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.get_dir_vec()
        rx, ry = self.get_right_vec()

        # Compute the absolute coordinates of the top-left view corner
        sz = AGENT_VIEW_SIZE
        hs = AGENT_VIEW_SIZE // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - AGENT_VIEW_SIZE // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - AGENT_VIEW_SIZE // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - AGENT_VIEW_SIZE + 1
            topY = self.agent_pos[1] - AGENT_VIEW_SIZE // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - AGENT_VIEW_SIZE // 2
            topY = self.agent_pos[1] - AGENT_VIEW_SIZE + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + AGENT_VIEW_SIZE
        botY = topY + AGENT_VIEW_SIZE

        return (topX, topY, botX, botY)

    def agent_sees(self, x, y):
        """
        Check if a grid position is visible to the agent
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= AGENT_VIEW_SIZE or vy >= AGENT_VIEW_SIZE:
            return False

        obs = self.gen_obs()
        obs_grid = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.agent_pos + self.get_dir_vec()

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = 1

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Wait/do nothing
        elif action == self.actions.wait:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, AGENT_VIEW_SIZE, AGENT_VIEW_SIZE)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(3, 6))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode()

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

    def get_obs_render(self, obs):
        """
        Render an agent observation for visualization
        """

        if self.obs_render == None:
            self.obs_render = Renderer(
                AGENT_VIEW_SIZE * CELL_PIXELS // 2,
                AGENT_VIEW_SIZE * CELL_PIXELS // 2
            )

        r = self.obs_render

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

    def render(self, mode='human', close=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None:
            self.grid_render = Renderer(
                self.grid_size * CELL_PIXELS,
                self.grid_size * CELL_PIXELS,
                True if mode == 'human' else False
            )

        r = self.grid_render

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, CELL_PIXELS)

        # Draw the agent
        r.push()
        r.translate(
            CELL_PIXELS * (self.agent_pos[0] + 0.5),
            CELL_PIXELS * (self.agent_pos[1] + 0.5)
        )
        r.rotate(self.agent_dir * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            ( 12,  0),
            (-12, -10)
        ])
        r.pop()

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the absolute coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.get_dir_vec()
        r_vec = self.get_right_vec()
        top_left = self.agent_pos + f_vec * (AGENT_VIEW_SIZE-1) - r_vec * (AGENT_VIEW_SIZE // 2)

        # For each cell in the visibility mask
        for vis_j in range(0, AGENT_VIEW_SIZE):
            for vis_i in range(0, AGENT_VIEW_SIZE):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                # Highlight the cell
                r.fillRect(
                    abs_i * CELL_PIXELS,
                    abs_j * CELL_PIXELS,
                    CELL_PIXELS,
                    CELL_PIXELS,
                    255, 255, 255, 75
                )

        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()

        return r
