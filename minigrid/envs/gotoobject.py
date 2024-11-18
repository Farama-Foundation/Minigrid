from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv


class GoToObjectEnv(MiniGridEnv):
    """
    ## Description

    This environment is a room with colored objects. The agent
    receives a textual (mission) string as input, telling it which colored object to go
    to, (eg: "go to the red key"). It receives a positive reward for performing
    the `done` action next to the correct object, as indicated in the mission
    string.

    ## Mission Space

    "go to the {color} {obj_type}"

    {color} is the color of the object. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".
    {obj_type} is the type of the object. Can be "key", "ball", "box".

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

    - `MiniGrid-GoToObject-6x6-N2-v0`
    - `MiniGrid-GoToObject-8x8-N2-v0`

    """

    def __init__(self, size=6, numObjs=2, max_steps: int | None = None, **kwargs):
        self.numObjs = numObjs
        self.size = size
        # Types of objects to be generated
        self.obj_types = ["key", "ball", "box"]

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, self.obj_types],
        )

        if max_steps is None:
            max_steps = 5 * size**2

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
    def _gen_mission(color: str, obj_type: str):
        return f"go to the {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Types and colors of objects we can generate
        types = ["key", "ball", "box"]

        objs = []
        objPos = []

        # Until we have generated all the objects
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            elif objType == "box":
                obj = Box(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key, ball and box.".format(
                        objType
                    )
                )

            pos = self.place_obj(obj)
            objs.append((objType, objColor))
            objPos.append(pos)

        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        objIdx = self._rand_int(0, len(objs))
        self.targetType, self.target_color = objs[objIdx]
        self.target_pos = objPos[objIdx]

        descStr = f"{self.target_color} {self.targetType}"
        self.mission = "go to the %s" % descStr
        # print(self.mission)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Toggle/pickup action terminates the episode
        if action == self.actions.toggle:
            terminated = True

        # Reward performing the done action next to the target object
        if action == self.actions.done:
            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info
