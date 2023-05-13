from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball, Box, Key


class ObstructedMazeEnv(RoomGrid):

    """
    ## Description

    The agent has to pick up a box which is placed in a corner of a 3x3 maze.
    The doors are locked, the keys are hidden in boxes and doors are obstructed
    by balls. This environment can be solved without relying on language.

    ## Mission Space

    "pick up the {COLOR_NAMES[0]} ball"

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
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent picks up the blue ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    "NDl" are the number of doors locked.
    "h" if the key is hidden in a box.
    "b" if the door is obstructed by a ball.
    "Q" number of quarters that will have doors and keys out of the 9 that the
    map already has.
    "Full" 3x3 maze with "h" and "b" options.
    "v1" prevents the key from being covered by the blocking ball. Only 2Dlhb, 1Q, 2Q, and Full are
    updated to v1. Other configurations won't face this issue because there is no blocking ball (1Dl,
    1Dlh, 2Dl, 2Dlh) or the only blocking ball is added before the key (1Dlhb).

    - `MiniGrid-ObstructedMaze-1Dl-v0`
    - `MiniGrid-ObstructedMaze-1Dlh-v0`
    - `MiniGrid-ObstructedMaze-1Dlhb-v0`
    - `MiniGrid-ObstructedMaze-2Dl-v0`
    - `MiniGrid-ObstructedMaze-2Dlh-v0`
    - `MiniGrid-ObstructedMaze-2Dlhb-v0`
    - `MiniGrid-ObstructedMaze-2Dlhb-v1`
    - `MiniGrid-ObstructedMaze-1Q-v0`
    - `MiniGrid-ObstructedMaze-1Q-v1`
    - `MiniGrid-ObstructedMaze-2Q-v0`
    - `MiniGrid-ObstructedMaze-2Q-v1`
    - `MiniGrid-ObstructedMaze-Full-v0`
    - `MiniGrid-ObstructedMaze-Full-v1`

    """

    def __init__(
        self,
        num_rows,
        num_cols,
        num_rooms_visited,
        max_steps: int | None = None,
        **kwargs,
    ):
        room_size = 6

        if max_steps is None:
            max_steps = 4 * num_rooms_visited * room_size**2

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[[COLOR_NAMES[0]]],
        )
        super().__init__(
            mission_space=mission_space,
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            max_steps=max_steps,
            **kwargs,
        )
        self.obj = Ball()  # initialize the obj attribute, that will be changed later on

    @staticmethod
    def _gen_mission(color: str):
        return f"pick up the {color} ball"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Define all possible colors for doors
        self.door_colors = self._rand_subset(COLOR_NAMES, len(COLOR_NAMES))
        # Define the color of the ball to pick up
        self.ball_to_find_color = COLOR_NAMES[0]
        # Define the color of the balls that obstruct doors
        self.blocking_ball_color = COLOR_NAMES[1]
        # Define the color of boxes in which keys are hidden
        self.box_color = COLOR_NAMES[2]

        self.mission = "pick up the %s ball" % self.ball_to_find_color

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True

        return obs, reward, terminated, truncated, info

    def add_door(
        self,
        i,
        j,
        door_idx=0,
        color=None,
        locked=False,
        key_in_box=False,
        blocked=False,
    ):
        """
        Add a door. If the door must be locked, it also adds the key.
        If the key must be hidden, it is put in a box. If the door must
        be obstructed, it adds a ball in front of the door.
        """

        door, door_pos = super().add_door(i, j, door_idx, color, locked=locked)

        if blocked:
            vec = DIR_TO_VEC[door_idx]
            blocking_ball = Ball(self.blocking_ball_color) if blocked else None
            self.grid.set(door_pos[0] - vec[0], door_pos[1] - vec[1], blocking_ball)

        if locked:
            obj = Key(door.color)
            if key_in_box:
                box = Box(self.box_color)
                box.contains = obj
                obj = box
            self.place_in_room(i, j, obj)

        return door, door_pos


class ObstructedMaze_1Dlhb(ObstructedMazeEnv):
    """
    A blue ball is hidden in a 2x1 maze. A locked door separates
    rooms. Doors are obstructed by a ball and keys are hidden in boxes.
    """

    def __init__(self, key_in_box=True, blocked=True, **kwargs):
        self.key_in_box = key_in_box
        self.blocked = blocked

        super().__init__(num_rows=1, num_cols=2, num_rooms_visited=2, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.add_door(
            0,
            0,
            door_idx=0,
            color=self.door_colors[0],
            locked=True,
            key_in_box=self.key_in_box,
            blocked=self.blocked,
        )

        self.obj, _ = self.add_object(1, 0, "ball", color=self.ball_to_find_color)
        self.place_agent(0, 0)


class ObstructedMaze_Full(ObstructedMazeEnv):
    """
    A blue ball is hidden in one of the 4 corners of a 3x3 maze. Doors
    are locked, doors are obstructed by a ball and keys are hidden in
    boxes.
    """

    def __init__(
        self,
        agent_room=(1, 1),
        key_in_box=True,
        blocked=True,
        num_quarters=4,
        num_rooms_visited=25,
        **kwargs,
    ):
        self.agent_room = agent_room
        self.key_in_box = key_in_box
        self.blocked = blocked
        self.num_quarters = num_quarters

        super().__init__(
            num_rows=3, num_cols=3, num_rooms_visited=num_rooms_visited, **kwargs
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        middle_room = (1, 1)
        # Define positions of "side rooms" i.e. rooms that are neither
        # corners nor the center.
        side_rooms = [(2, 1), (1, 2), (0, 1), (1, 0)][: self.num_quarters]
        for i in range(len(side_rooms)):
            side_room = side_rooms[i]

            # Add a door between the center room and the side room
            self.add_door(
                *middle_room, door_idx=i, color=self.door_colors[i], locked=False
            )

            for k in [-1, 1]:
                # Add a door to each side of the side room
                self.add_door(
                    *side_room,
                    locked=True,
                    door_idx=(i + k) % 4,
                    color=self.door_colors[(i + k) % len(self.door_colors)],
                    key_in_box=self.key_in_box,
                    blocked=self.blocked,
                )

        corners = [(2, 0), (2, 2), (0, 2), (0, 0)][: self.num_quarters]
        ball_room = self._rand_elem(corners)

        self.obj, _ = self.add_object(
            ball_room[0], ball_room[1], "ball", color=self.ball_to_find_color
        )
        self.place_agent(*self.agent_room)


class ObstructedMaze_2Dl(ObstructedMaze_Full):
    def __init__(self, **kwargs):
        super().__init__((2, 1), False, False, 1, 4, **kwargs)


class ObstructedMaze_2Dlh(ObstructedMaze_Full):
    def __init__(self, **kwargs):
        super().__init__((2, 1), True, False, 1, 4, **kwargs)


class ObstructedMaze_2Dlhb(ObstructedMaze_Full):
    def __init__(self, **kwargs):
        super().__init__((2, 1), True, True, 1, 4, **kwargs)
