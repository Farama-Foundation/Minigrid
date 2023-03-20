from __future__ import annotations

from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball, Box, Key
from minigrid.envs.obstructedmaze import ObstructedMazeEnv


class ObstructedMaze_Full(ObstructedMazeEnv):
    """
    A blue ball is hidden in one of the 4 corners of a 3x3 maze. Doors
    are locked, doors are obstructed by a ball and keys are hidden in
    boxes.

    All doors and their corresponding blocking balls will be added first,
    followed by the boxes containing the keys.
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
                # Add a door to each side of the side room w/o placing a key
                self.add_locked_door(
                    *side_room,
                    door_idx=(i + k) % 4,
                    color=self.door_colors[(i + k) % len(self.door_colors)],
                    blocked=self.blocked,
                )

            # Add keys after all doors and their blocking balls are added
            for k in [-1, 1]:
                self.add_key(
                    *side_room,
                    color=self.door_colors[(i + k) % len(self.door_colors)],
                    key_in_box=self.key_in_box,
                )

        corners = [(2, 0), (2, 2), (0, 2), (0, 0)][: self.num_quarters]
        ball_room = self._rand_elem(corners)

        self.obj, _ = self.add_object(
            ball_room[0], ball_room[1], "ball", color=self.ball_to_find_color
        )
        self.place_agent(*self.agent_room)

    def add_locked_door(self, i, j, door_idx=0, color=None, blocked=False):
        door, door_pos = RoomGrid.add_door(self, i, j, door_idx, color, locked=True)

        if blocked:
            vec = DIR_TO_VEC[door_idx]
            blocking_ball = Ball(self.blocking_ball_color) if blocked else None
            self.grid.set(door_pos[0] - vec[0], door_pos[1] - vec[1], blocking_ball)

        return door, door_pos

    def add_key(
        self,
        i,
        j,
        color=None,
        key_in_box=False,
    ):
        obj = Key(color)
        if key_in_box:
            box = Box(self.box_color)
            box.contains = obj
            obj = box
        self.place_in_room(i, j, obj)
