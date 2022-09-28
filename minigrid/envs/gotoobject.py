from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv


class GoToObjectEnv(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, size=6, numObjs=2, **kwargs):
        self.numObjs = numObjs
        self.size = size
        # Types of objects to be generated
        self.obj_types = ["key", "ball", "box"]

        mission_space = MissionSpace(
            mission_func=lambda color, type: f"go to the {color} {type}",
            ordered_placeholders=[COLOR_NAMES, self.obj_types],
        )
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=5 * size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs,
        )

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
            if abs(ax - tx) <= 1 and abs(ay - ty) <= 1:
                reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info
