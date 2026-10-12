from __future__ import annotations

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor, Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv


class MazeEnv(MiniGridEnv):
    """
    ## Description

    A randomly generated perfect maze. The agent starts in a passage along the
    left edge and must reach a green goal along the right edge. The randomized
    Prim-style generator grows connected passages by opening a frontier wall
    only when it has exactly one adjacent passage, preventing cycles.

    Interior obstacles are lava by default, or walls with `obstacle_type=Wall`.
    The outer boundary is always walls. Resetting with the same seed reproduces
    the maze, entrance and goal. The default observation is partial; walls block
    sight and lava retains MiniGrid's usual transparent visibility behavior.
    `FullyObsWrapper` exposes the full grid.

    ![Example wall and lava mazes](../../_static/images/maze_examples.png)

    ## Baselines

    Across maze seeds 0-99, a full-map breadth-first planner solved all six
    configurations. Mean action counts (including turns) were 15.47, 21.41 and
    32.36 for sizes 9, 11 and 15. Uniformly sampling all seven actions with a
    NumPy RNG seeded by the maze seed gave wall success rates of 3%, 4% and 0%,
    and lava rates of 0%, 1% and 0%, within the default budget. These are planning
    and random-policy baselines, not partial-observation training results.

    ## Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    Success gives `1 - 0.9 * step_count / max_steps`; failure gives zero.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    `size` must be an odd integer of at least 5. The default step budget is
    `4 * size**2`; smaller custom budgets may prevent completion.

    - `MiniGrid-MazeS9-v0`
    - `MiniGrid-MazeS11-v0`
    - `MiniGrid-MazeS15-v0`
    - `MiniGrid-MazeLavaS9-v0`
    - `MiniGrid-MazeLavaS11-v0`
    - `MiniGrid-MazeLavaS15-v0`

    """

    def __init__(
        self,
        size=9,
        obstacle_type=Lava,
        max_steps: int | None = None,
        **kwargs,
    ):
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 5
            or size % 2 == 0
        ):
            raise ValueError("size must be an odd integer of at least 5")
        if isinstance(obstacle_type, str):
            obstacle_type = {"wall": Wall, "lava": Lava}.get(obstacle_type)
        if obstacle_type not in (Wall, Lava):
            raise ValueError("obstacle_type must be Wall, Lava, 'wall', or 'lava'")
        self.obstacle_type = obstacle_type

        if obstacle_type == Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        # A temporary, valid grid object distinguishes passages from unvisited cells.
        cell = Floor()

        starting_x = self._rand_int(1, width - 1)
        starting_y = self._rand_int(1, height - 1)

        self.grid.set(starting_x, starting_y, cell)

        walls = []
        for i, j in ([-1, 0], [0, -1], [0, 1], [1, 0]):
            x, y = starting_x + i, starting_y + j
            if 0 < x < width - 1 and 0 < y < height - 1:
                walls.append([x, y])
                self.put_obj(self.obstacle_type(), x, y)

        # Find number of surrounding cells
        def surroundingCells(rand_wall):
            s_cells = 0
            for i, j in ([-1, 0], [0, -1], [0, 1], [1, 0]):
                if self.grid.get(rand_wall[0] + i, rand_wall[1] + j) == cell:
                    s_cells += 1
            return s_cells

        def delete_wall(walls, rand_wall):
            walls.remove(rand_wall)

        def mark(walls, cell, x, y):
            if 0 < x < width - 1 and 0 < y < height - 1 and self.grid.get(x, y) != cell:
                self.put_obj(self.obstacle_type(), x, y)
                if [x, y] not in walls:
                    walls.append([x, y])

        def helper(rand_wall, cell, wall1, wall2, wall3):
            # Find the number of surrounding cells
            s_cells = surroundingCells(rand_wall)
            if s_cells == 1:
                # Denote the new path
                self.grid.set(rand_wall[0], rand_wall[1], cell)

                # Mark the new walls
                mark(*wall1)

                mark(*wall2)

                mark(*wall3)

        while walls:
            # Pick a random wall
            rand_wall = walls[self._rand_int(0, len(walls))]

            # Upper cell
            up = (walls, cell, rand_wall[0], rand_wall[1] - 1)

            # Bottom cell
            bot = (walls, cell, rand_wall[0], rand_wall[1] + 1)

            # Leftmost cell
            left = (walls, cell, rand_wall[0] - 1, rand_wall[1])

            # Rightmost cell
            right = (walls, cell, rand_wall[0] + 1, rand_wall[1])

            # Check if it is a left wall
            if rand_wall[0] != 0:

                if self.grid.get(rand_wall[0] + 1, rand_wall[1]) == cell:

                    helper(rand_wall, cell, up, bot, left)

                    # Delete wall
                    delete_wall(walls, rand_wall)

                    continue

            # Check if it is an upper wall
            if rand_wall[1] != 0:
                if self.grid.get(rand_wall[0], rand_wall[1] + 1) == cell:

                    helper(rand_wall, cell, up, left, right)

                    # Delete wall
                    delete_wall(walls, rand_wall)

                    continue

            # Check the bottom wall
            if rand_wall[1] != height - 1:
                if self.grid.get(rand_wall[0], rand_wall[1] - 1) == cell:

                    helper(rand_wall, cell, bot, left, right)

                    # Delete wall
                    delete_wall(walls, rand_wall)

                    continue

            # Check the right wall
            if rand_wall[0] != width - 1:
                if self.grid.get(rand_wall[0] - 1, rand_wall[1]) == cell:

                    helper(rand_wall, cell, right, bot, up)

                    # Delete wall
                    delete_wall(walls, rand_wall)

                    continue

            # Delete the wall from the list anyway
            delete_wall(walls, rand_wall)

        # Set entrance and exit
        cells = []
        for i in range(1, height - 1):
            if self.grid.get(1, i) == cell:
                cells.append((1, i))

        self.agent_pos = np.array(self.np_random.choice(cells))
        self.agent_dir = 0

        cells = []
        for i in range(height - 2, 0, -1):
            if self.grid.get(width - 2, i) == cell:
                cells.append((width - 2, i))

        pt = self.np_random.choice(cells)
        self.put_obj(Goal(), pt[0], pt[1])

        # Mark the remaining unvisited cells as walls
        for i in range(0, width):
            for j in range(0, height):
                if self.grid.get(i, j) is None:
                    self.put_obj(self.obstacle_type(), i, j)
                if self.grid.get(i, j) == cell:
                    self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
