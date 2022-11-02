import hashlib
import math
from abc import abstractmethod
from enum import IntEnum
from typing import Any, Callable, Optional, Union

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

# Size in pixels of a tile in the full-scale human view
from gym_minigrid.minigrid import Grid, TILE_PIXELS, DIR_TO_VEC, WorldObj, Wall, OBJECT_TO_IDX
from gym_minigrid.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)
from gym_minigrid.window import Window
from gym_minigrid.rule import extract_ruleset


def rand_int(low, high):
    """
    Generate random integer in [low,high[
    """
    # TODO: seed
    # return np_random.integers(low, high)
    return np.random.randint(low, high)


class BabaIsYouGrid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        # self.grid = [[None]] * width * height  # self.grid[0].append(...) modifies all the elements and not just the first one
        self.grid = [[None] for _ in range(width * height)]

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
        idx = j * self.width + i

        if v is None:
            if self.grid[idx] == [None]:
                self.grid[idx] = [None]
            else:
                # remove the obj at the top
                self.grid[idx].pop()
        else:
            # stack objects
            self.grid[idx].append(v)

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height

        return self.grid[j * self.width + i][-1]

    def __iter__(self):
        for elem in self.grid.__iter__():
            yield elem[-1]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    @classmethod
    def render_tile(
        cls, obj, agent_dir=None, highlight=False, tile_size=TILE_PIXELS, subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        # if agent_dir is not None:
        #     tri_fn = point_in_triangle(
        #         (0.12, 0.19),
        #         (0.87, 0.50),
        #         (0.12, 0.81),
        #     )
        #
        #     # Rotate the agent based on its direction
        #     tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
        #     fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, tile_size, agent_pos=None, agent_dir=None, highlight_mask=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = BabaIsYouGrid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX["empty"]
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = type_idx != OBJECT_TO_IDX["unseen"]

        return grid, vis_mask

    def process_vis(self, agent_pos):
        mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, self.height)):
            for i in range(0, self.width - 1):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, self.width)):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, self.height):
            for i in range(0, self.width):
                if not mask[i, j]:
                    self.set(i, j, None)

        return mask


class BabaIsYouEnv(gym.Env):
    metadata = {
        # Deprecated: use 'render_modes' instead
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,  # Deprecated: use 'render_fps' instead
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 10,
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        idle = 0
        up = 1
        right = 2
        down = 3
        left = 4

    def __init__(
        self,
        grid_size: int = None,
        width: int = None,
        height: int = None,
        max_steps: int = 100,
        **kwargs,
    ):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = BabaIsYouEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(width, height, 3),
            dtype="uint8",
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.window: Window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps

        # Current position and direction of the agent
        self.agent_pos: np.ndarray = None
        self.agent_dir: int = None
        self.agent_object = 'baba'

        # Initialize the state
        self.grid = None
        self.reset()

        self._ruleset = {}

    def get_ruleset(self):
        return self._ruleset

    def reset(self, *, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # Compute the ruleset for the generated grid
        self._ruleset = extract_ruleset(self.grid)
        # make the ruleset accessible to all FlexibleWorlObj (not working for objects added after reset is called)
        for e in self.grid:
            if hasattr(e, "set_ruleset_getter"):
                e.set_ruleset_getter(self.get_ruleset)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.grid.encode()

        if not return_info:
            return obs
        else:
            return obs, {}

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()
        # TODO: make the agent part of the grid encoding
        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        str = ""

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c is None:
                    str += "  "
                    continue

                if c.type == "door":
                    if c.is_open:
                        str += "__"
                    elif c.is_locked:
                        str += "L" + c.color[0].upper()
                    else:
                        str += "D" + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += "\n"

        return str

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = np.array(
                (
                    rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                )
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None
        # pos = self.place_obj(None, top, size, max_tries=max_tries)
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = rand_int(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    def change_obj_pos(self, pos, new_pos, mvt_dir=None):
        """
        Change the position of an object in the grid
        """
        if np.any(pos != new_pos):
            # move the object
            e = self.grid.get(*pos)
            self.grid.set(*new_pos, e)
            self.grid.set(*pos, None)
            # change the dir of the object
            if mvt_dir is not None:
                e.dir = np.argwhere(np.all(DIR_TO_VEC == mvt_dir, axis=1))[0][0]

    def is_win_pos(self, pos):
        new_cell = self.grid.get(*pos)
        return new_cell is not None and new_cell.is_goal()

    def is_lose_pos(self, pos):
        new_cell = self.grid.get(*pos)
        return new_cell is not None and new_cell.is_defeat()

    def move(self, pos, dir_vec):
        """
        Return fwd_pos if can move, otherwise return pos
        """
        # TODO: win only if the agent is on a winning block? Win and lose rules apply only to the agent, not to the pushed objects
        # if the agent pushes an obj on a winning block, win the game but if it is a losing block, just destroy the obj
        # is_obj_win = False

        fwd_pos = pos + dir_vec
        # if fwd_cell can be pushed, try to move it
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is not None and fwd_cell.can_push():
            # new_fwd_pos, is_obj_win, is_obj_lose = self.move(fwd_pos, dir_vec)
            new_fwd_pos, _, _ = self.move(fwd_pos, dir_vec)
            # TODO: hot and melt rules
            # if is_obj_lose:
            #     self.grid.set(*fwd_pos, None)  # destroy the obj
            # else:
            # self.change_obj_pos(fwd_pos, new_fwd_pos)  # move the obj

        # move if the fwd cell is empty or can overlap
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None or fwd_cell.can_overlap():
            new_pos = tuple(fwd_pos)
        else:
            new_pos = pos

        # TODO: move the agent here or just compute the mvts and execute them later?
        # check if win or lose before moving the agent
        is_win = self.is_win_pos(new_pos)
        is_lose = self.is_lose_pos(new_pos)
        self.change_obj_pos(pos, new_pos, dir_vec)

        # pull object in the cell behind
        bwd_pos = pos - dir_vec
        bwd_cell = self.grid.get(*bwd_pos)
        # the obj can be pulled if the agent has moved (i.e. new_pos = fwd_pos)
        if bwd_cell is not None and bwd_cell.is_pull():
            new_bwd_pos, _, _ = self.move(bwd_pos, dir_vec)
            # self.change_obj_pos(bwd_pos, new_bwd_pos)

        return new_pos, is_win, is_lose

    def step(self, action):
        self.step_count += 1

        is_win, is_lose = False, False
        reward = 0
        done = False

        if action == self.actions.up:
            self.agent_dir = 3
        elif action == self.actions.right:
            self.agent_dir = 0
        elif action == self.actions.down:
            self.agent_dir = 1
        elif action == self.actions.left:
            self.agent_dir = 2

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        if action != self.actions.idle:
            # move the agent if the forward cell is empty or can overlap or can be pushed
            # self.agent_pos, is_win, is_lose = self.move(self.agent_pos, self.dir_vec)

            for k, e in enumerate(self.grid):
                if e is not None and (e.is_agent() or e.is_move()):
                    e.has_moved = False

            # movements = []
            # the agent moves first
            for k, e in enumerate(self.grid):
                if e is not None and e.is_agent() and not e.has_moved:
                    e.dir = self.agent_dir
                    pos = (k % self.grid.width, k // self.grid.width)
                    new_pos, is_win, is_lose = self.move(pos, self.dir_vec)
                    # movements.append((pos, new_pos))
                    e.has_moved = True

            # move other objects
            for k, e in enumerate(self.grid):
                if e is not None and e.is_move() and not e.has_moved:
                    pos = (k % self.grid.width, k // self.grid.width)
                    new_pos, _, _ = self.move(pos, DIR_TO_VEC[e.dir])
                    e.has_moved = True

            # TODO: handle conflicts
            # for (pos, new_pos) in movements:
            #     self.change_obj_pos(pos, new_pos)

            if is_win:
                done = True
                reward = self._reward()
            elif is_lose:
                done = True
                reward = -1

            self._ruleset = extract_ruleset(self.grid)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.grid.encode()

        return obs, reward, done, {}

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        return img

    def render(self, mode="human", highlight=True, tile_size=TILE_PIXELS):
        assert mode in self.metadata["render_modes"]
        """
        Render the whole-grid human view
        """
        if mode == "human" and not self.window:
            self.window = Window("gym_minigrid")
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir
        )

        if mode == "human":
            # self.window.set_caption(self.mission)
            self.window.show_img(img)
        else:
            return img

    def close(self):
        if self.window:
            self.window.close()
