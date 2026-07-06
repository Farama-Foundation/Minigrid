from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class LockedRoom:
    def __init__(self, top, size, doorPos):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)


class LockedRoomEnv(MiniGridEnv):
    """
    ## Description

    The environment has six rooms, one of which is locked. The agent receives
    a textual mission string as input, telling it which room to go to in order
    to get the key that opens the locked room. It then has to go into the locked
    room in order to reach the final goal. This environment is extremely
    difficult to solve with vanilla reinforcement learning alone.

    ## Mission Space

    "get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal"

    {lockedroom_color}, {keyroom_color}, and {door_color} can be "red", "green",
    "blue", "purple", "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

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

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-LockedRoom-v0`

    """

    def __init__(self, size=19, max_steps: int | None = None, **kwargs):
        self.size = size

        if max_steps is None:
            max_steps = 10 * size
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES] * 3,
        )
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(lockedroom_color: str, keyroom_color: str, door_color: str):
        return (
            f"get the {lockedroom_color} key from the {keyroom_color} room,"
            f" unlock the {door_color} door and go to the goal"
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        for i in range(0, width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 1, Wall())
        for j in range(0, height):
            self.grid.set(0, j, Wall())
            self.grid.set(width - 1, j, Wall())

        # Hallway walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        for j in range(0, height):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

        self.rooms = []

        # Room splitting walls
        for n in range(0, 3):
            j = n * (height // 3)
            for i in range(0, lWallIdx):
                self.grid.set(i, j, Wall())
            for i in range(rWallIdx, width):
                self.grid.set(i, j, Wall())

            roomW = lWallIdx + 1
            roomH = height // 3 + 1
            self.rooms.append(LockedRoom((0, j), (roomW, roomH), (lWallIdx, j + 3)))
            self.rooms.append(
                LockedRoom((rWallIdx, j), (roomW, roomH), (rWallIdx, j + 3))
            )

        # Choose one random room to be locked
        lockedRoom = self._rand_elem(self.rooms)
        lockedRoom.locked = True
        goalPos = lockedRoom.rand_pos(self)
        self.grid.set(*goalPos, Goal())

        # Assign the door colors
        colors = set(COLOR_NAMES)
        for room in self.rooms:
            color = self._rand_elem(sorted(colors))
            colors.remove(color)
            room.color = color
            if room.locked:
                self.grid.set(*room.doorPos, Door(color, is_locked=True))
            else:
                self.grid.set(*room.doorPos, Door(color))

        # Select a random room to contain the key
        while True:
            keyRoom = self._rand_elem(self.rooms)
            if keyRoom != lockedRoom:
                break
        keyPos = keyRoom.rand_pos(self)
        self.grid.set(*keyPos, Key(lockedRoom.color))

        # Randomize the player start position and orientation
        self.agent_pos = self.place_agent(
            top=(lWallIdx, 0), size=(rWallIdx - lWallIdx, height)
        )

        # Generate the mission string
        self.mission = (
            "get the %s key from the %s room, "
            "unlock the %s door and "
            "go to the goal"
        ) % (lockedRoom.color, keyRoom.color, lockedRoom.color)
