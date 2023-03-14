from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key
from minigrid.minigrid_env import MiniGridEnv


class FetchEnv(MiniGridEnv):

    """
    ## Description

    This environment has multiple objects of assorted types and colors. The
    agent receives a textual string as part of its observation telling it which
    object to pick up. Picking up the wrong object terminates the episode with
    zero reward.

    ## Mission Space

    "{syntax} {color} {type}"

    {syntax} is one of the following: "get a", "go get a", "fetch a",
    "go fetch a", "you must fetch a".

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "key" or "ball".

    ## Action Space

    | Num | Name         | Action               |
    |-----|--------------|----------------------|
    | 0   | left         | Turn left            |
    | 1   | right        | Turn right           |
    | 2   | forward      | Move forward         |
    | 3   | pickup       | Pick up an object    |
    | 4   | drop         | Unused               |
    | 5   | toggle       | Unused               |
    | 6   | done         | Unused               |

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

    1. The agent picks up the correct object.
    2. The agent picks up the wrong object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    N: number of objects to be generated.

    - `MiniGrid-Fetch-5x5-N2-v0`
    - `MiniGrid-Fetch-6x6-N2-v0`
    - `MiniGrid-Fetch-8x8-N3-v0`

    """

    def __init__(self, size=8, numObjs=3, max_steps: int | None = None, **kwargs):
        self.numObjs = numObjs
        self.obj_types = ["key", "ball"]

        MISSION_SYNTAX = [
            "get a",
            "go get a",
            "fetch a",
            "go fetch a",
            "you must fetch a",
        ]
        self.size = size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[MISSION_SYNTAX, COLOR_NAMES, self.obj_types],
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
    def _gen_mission(syntax: str, color: str, obj_type: str):
        return f"{syntax} {color} {obj_type}"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objType = self._rand_elem(self.obj_types)
            objColor = self._rand_elem(COLOR_NAMES)

            if objType == "key":
                obj = Key(objColor)
            elif objType == "ball":
                obj = Ball(objColor)
            else:
                raise ValueError(
                    "{} object type given. Object type can only be of values key and ball.".format(
                        objType
                    )
                )

            self.place_obj(obj)
            objs.append(obj)

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = f"{self.targetColor} {self.targetType}"

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = "get a %s" % descStr
        elif idx == 1:
            self.mission = "go get a %s" % descStr
        elif idx == 2:
            self.mission = "fetch a %s" % descStr
        elif idx == 3:
            self.mission = "go fetch a %s" % descStr
        elif idx == 4:
            self.mission = "you must fetch a %s" % descStr
        assert hasattr(self, "mission")

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.carrying:
            if (
                self.carrying.color == self.targetColor
                and self.carrying.type == self.targetType
            ):
                reward = self._reward()
                terminated = True
            else:
                reward = 0
                terminated = True

        return obs, reward, terminated, truncated, info
