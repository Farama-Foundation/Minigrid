from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Key
from minigrid.minigrid_env import MiniGridEnv


class PlaygroundEnv(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.
    """

    def __init__(self, max_steps=100, **kwargs):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.size = 19
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return ""

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        roomW = width // 3
        roomH = height // 3

        # For each row of rooms
        for j in range(0, 3):
            # For each column
            for i in range(0, 3):
                xL = i * roomW
                yT = j * roomH
                xR = xL + roomW
                yB = yT + roomH

                # Bottom wall and door
                if i + 1 < 3:
                    self.grid.vert_wall(xR, yT, roomH)
                    pos = (xR, self._rand_int(yT + 1, yB - 1))
                    color = self._rand_elem(COLOR_NAMES)
                    self.grid.set(*pos, Door(color))

                # Bottom wall and door
                if j + 1 < 3:
                    self.grid.horz_wall(xL, yB, roomW)
                    pos = (self._rand_int(xL + 1, xR - 1), yB)
                    color = self._rand_elem(COLOR_NAMES)
                    self.grid.set(*pos, Door(color))

        # Randomize the player start position and orientation
        self.place_agent()

        # Place random objects in the world
        types = ["key", "ball", "box"]
        for i in range(0, 12):
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)
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
            self.place_obj(obj)

        # No explicit mission in this environment
        self.mission = ""
