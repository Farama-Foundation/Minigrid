from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door
from minigrid.minigrid_env import MiniGridEnv


class GoToDoorEnv(MiniGridEnv):
    """
    ## Description

    This environment is a room with four doors, one on each wall. The agent
    receives a textual (mission) string as input, telling it which door to go
    to, (eg: "go to the red door"). It receives a positive reward for performing
    the `done` action next to the correct door, as indicated in the mission
    string.

    ## Mission Space

    "go to the {color} door"

    {color} is the color of the door. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action               |
    |-----|--------------|----------------------|
    | 0   | left         | Turn left            |
    | 1   | right        | Turn right           |
    | 2   | forward      | Move forward         |
    | 3   | pickup       | Unused               |
    | 4   | drop         | Unused               |
    | 5   | toggle       | Unused               |
    | 6   | done         | Done completing task |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent stands next the correct door performing the `done` action.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-GoToDoor-5x5-v0`
    - `MiniGrid-GoToDoor-6x6-v0`
    - `MiniGrid-GoToDoor-8x8-v0`

    """

    def __init__(self, size=5, max_steps: int | None = None, **kwargs):
        assert size >= 5
        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES],
        )

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str):
        return f"go to the {color} door"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Randomly vary the room width and height
        width = self._rand_int(5, width + 1)
        height = self._rand_int(5, height + 1)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the 4 doors at random positions
        doorPos = []
        doorPos.append((self._rand_int(2, width - 2), 0))
        doorPos.append((self._rand_int(2, width - 2), height - 1))
        doorPos.append((0, self._rand_int(2, height - 2)))
        doorPos.append((width - 1, self._rand_int(2, height - 2)))

        # Generate the door colors
        doorColors = []
        while len(doorColors) < len(doorPos):
            color = self._rand_elem(COLOR_NAMES)
            if color in doorColors:
                continue
            doorColors.append(color)

        # Place the doors in the grid
        for idx, pos in enumerate(doorPos):
            color = doorColors[idx]
            self.grid.set(*pos, Door(color))

        # Randomize the agent start position and orientation
        self.place_agent(size=(width, height))

        # Select a random target door
        doorIdx = self._rand_int(0, len(doorPos))
        self.target_pos = doorPos[doorIdx]
        self.target_color = doorColors[doorIdx]

        # Generate the mission string
        self.mission = "go to the %s door" % self.target_color

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Don't let the agent open any of the doors
        if action == self.actions.toggle:
            terminated = True

        # Reward performing done action in front of the target door
        if action == self.actions.done:
            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info
