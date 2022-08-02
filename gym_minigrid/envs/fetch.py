from gym_minigrid.minigrid import COLOR_NAMES, Ball, Grid, Key, MiniGridEnv
from gym_minigrid.register import register


class FetchEnv(MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings
    """

    def __init__(self, size=8, numObjs=3, **kwargs):
        self.numObjs = numObjs

        super().__init__(
            grid_size=size,
            max_steps=5 * size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        types = ["key", "ball"]

        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objType = self._rand_elem(types)
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
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.carrying:
            if (
                self.carrying.color == self.targetColor
                and self.carrying.type == self.targetType
            ):
                reward = self._reward()
                done = True
            else:
                reward = 0
                done = True

        return obs, reward, done, info


register(
    id="MiniGrid-Fetch-5x5-N2-v0",
    entry_point="gym_minigrid.envs.fetch:FetchEnv",
    size=5,
    numObjs=2,
)

register(
    id="MiniGrid-Fetch-6x6-N2-v0",
    entry_point="gym_minigrid.envs.fetch:FetchEnv",
    size=6,
    numObjs=2,
)

register(id="MiniGrid-Fetch-8x8-N3-v0", entry_point="gym_minigrid.envs.fetch:FetchEnv")
