from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv


class RedBlueDoorEnv(MiniGridEnv):

    """
    ## Description

    The agent is randomly placed within a room with one red and one blue door
    facing opposite directions. The agent has to open the red door and then open
    the blue door, in that order. Note that, surprisingly, this environment is
    solvable without memory.

    ## Mission Space

    "open the red door then the blue door"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the blue door having already opened the red door.
    2. The agent opens the blue door without having opened the red door yet.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-RedBlueDoors-6x6-v0`
    - `MiniGrid-RedBlueDoors-8x8-v0`

    """

    def __init__(self, size=8, max_steps: int | None = None, **kwargs):
        self.size = size
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 20 * size**2

        super().__init__(
            mission_space=mission_space,
            width=2 * size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "open the red door then the blue door"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        self.grid.wall_rect(0, 0, 2 * self.size, self.size)
        self.grid.wall_rect(self.size // 2, 0, self.size, self.size)

        # Place the agent in the top-left corner
        self.place_agent(top=(self.size // 2, 0), size=(self.size, self.size))

        # Add a red door at a random position in the left wall
        pos = self._rand_int(1, self.size - 1)
        self.red_door = Door("red")
        self.grid.set(self.size // 2, pos, self.red_door)

        # Add a blue door at a random position in the right wall
        pos = self._rand_int(1, self.size - 1)
        self.blue_door = Door("blue")
        self.grid.set(self.size // 2 + self.size - 1, pos, self.blue_door)

        # Generate the mission string
        self.mission = "open the red door then the blue door"

    def step(self, action):
        red_door_opened_before = self.red_door.is_open
        blue_door_opened_before = self.blue_door.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        red_door_opened_after = self.red_door.is_open
        blue_door_opened_after = self.blue_door.is_open

        if blue_door_opened_after:
            if red_door_opened_before:
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        elif red_door_opened_after:
            if blue_door_opened_before:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info
