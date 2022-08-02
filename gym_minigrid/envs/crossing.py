import itertools as itt

import numpy as np

from gym_minigrid.minigrid import Goal, Grid, Lava, MiniGridEnv, Wall
from gym_minigrid.register import register


class CrossingEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(self, size=9, num_crossings=1, obstacle_type=Lava, **kwargs):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=False,
            **kwargs
        )

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[: self.num_crossings]  # sample random rivers
        rivers_v = sorted(pos for direction, pos in rivers if direction is v)
        rivers_h = sorted(pos for direction, pos in rivers if direction is h)
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1])
                )
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1])
                )
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )


register(
    id="MiniGrid-LavaCrossingS9N1-v0",
    entry_point="gym_minigrid.envs.crossing:CrossingEnv",
    size=9,
    num_crossings=1,
)

register(
    id="MiniGrid-LavaCrossingS9N2-v0",
    entry_point="gym_minigrid.envs.crossing:CrossingEnv",
    size=9,
    num_crossings=2,
)

register(
    id="MiniGrid-LavaCrossingS9N3-v0",
    entry_point="gym_minigrid.envs.crossing:CrossingEnv",
    size=9,
    num_crossings=3,
)

register(
    id="MiniGrid-LavaCrossingS11N5-v0",
    entry_point="gym_minigrid.envs.crossing:CrossingEnv",
    size=11,
    num_crossings=5,
)

register(
    id="MiniGrid-SimpleCrossingS9N1-v0",
    entry_point="gym_minigrid.envs.crossing:CrossingEnv",
    size=9,
    num_crossings=1,
    obstacle_type=Wall,
)

register(
    id="MiniGrid-SimpleCrossingS9N2-v0",
    entry_point="gym_minigrid.envs.crossing:CrossingEnv",
    size=9,
    num_crossings=2,
    obstacle_type=Wall,
)

register(
    id="MiniGrid-SimpleCrossingS9N3-v0",
    entry_point="gym_minigrid.envs.crossing:CrossingEnv",
    size=9,
    num_crossings=3,
    obstacle_type=Wall,
)

register(
    id="MiniGrid-SimpleCrossingS11N5-v0",
    entry_point="gym_minigrid.envs.crossing:CrossingEnv",
    size=11,
    num_crossings=5,
    obstacle_type=Wall,
)
