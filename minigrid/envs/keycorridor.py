from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid


class KeyCorridorEnv(RoomGrid):
    """
    ## Description

    This environment is similar to the locked room environment, but there are
    multiple registered environment configurations of increasing size,
    making it easier to use curriculum learning to train an agent to solve it.
    The agent has to pick up an object which is behind a locked door. The key is
    hidden in another room, and the agent has to explore the environment to find
    it. The mission string does not give the agent any clues as to where the
    key is placed. This environment can be solved without relying on language.

    ## Mission Space

    "pick up the {color} {obj_type}"

    {color} is the color of the object. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "ball" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

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

    1. The agent picks up the correct object.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    S: room size.
    R: Number of rows.

    - `MiniGrid-KeyCorridorS3R1-v0`
    - `MiniGrid-KeyCorridorS3R2-v0`
    - `MiniGrid-KeyCorridorS3R3-v0`
    - `MiniGrid-KeyCorridorS4R3-v0`
    - `MiniGrid-KeyCorridorS5R3-v0`
    - `MiniGrid-KeyCorridorS6R3-v0`

    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.obj_type = obj_type
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, [obj_type]],
        )

        if max_steps is None:
            max_steps = 30 * room_size**2

        super().__init__(
            mission_space=mission_space,
            room_size=room_size,
            num_rows=num_rows,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), "key", door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
