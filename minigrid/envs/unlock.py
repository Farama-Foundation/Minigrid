from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid


class UnlockEnv(RoomGrid):

    """
    ### Description

    The agent has to open a locked door. This environment can be solved without
    relying on language.

    ### Mission Space

    "open the door"

    ### Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-Unlock-v0`

    """

    def __init__(self, **kwargs):
        room_size = 6
        mission_space = MissionSpace(mission_func=lambda: "open the door")
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8 * room_size**2,
            **kwargs
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.door = door
        self.mission = "open the door"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info
