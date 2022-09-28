import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv


class LavaGapEnv(MiniGridEnv):

    """
    ### Description

    The agent has to reach the green goal square at the opposite corner of the
    room, and must pass through a narrow gap in a vertical strip of deadly lava.
    Touching the lava terminate the episode with a zero reward. This environment
    is useful for studying safety and safe exploration.

    ### Mission Space

    Depending on the `obstacle_type` parameter:
    - `Lava`: "avoid the lava and get to the green goal square"
    - otherwise: "find the opening and get to the green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

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

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ### Registered Configurations

    S: size of map SxS.

    - `MiniGrid-LavaGapS5-v0`
    - `MiniGrid-LavaGapS6-v0`
    - `MiniGrid-LavaGapS7-v0`

    """

    def __init__(self, size, obstacle_type=Lava, **kwargs):
        self.obstacle_type = obstacle_type
        self.size = size

        if obstacle_type == Lava:
            mission_space = MissionSpace(
                mission_func=lambda: "avoid the lava and get to the green goal square"
            )
        else:
            mission_space = MissionSpace(
                mission_func=lambda: "find the opening and get to the green goal square"
            )

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=False,
            **kwargs
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # Generate and store random gap position
        self.gap_pos = np.array(
            (
                self._rand_int(2, width - 2),
                self._rand_int(1, height - 1),
            )
        )

        # Place the obstacle wall
        self.grid.vert_wall(self.gap_pos[0], 1, height - 2, self.obstacle_type)

        # Put a hole in the wall
        self.grid.set(*self.gap_pos, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
