from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random
from operator import add

class DynamicObstaclesEnv(MiniGridEnv):
    """
    Empty grid environment with moving obstacles
    """

    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
            self.start_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        n_obstacles = 10
        self.obstacles = []
        for i_obst in range(n_obstacles):
            y = random.randint(1, width - 2)
            x = random.randint(1, height - 2)
            while (x, y) == self.agent_start_pos or (x, y) == (height - 2, width - 2):
                y = random.randint(1, width - 2)
                x = random.randint(1, height - 2)
            self.obstacles.append(Obstacle())
            self.obstacles[i_obst].cur_pos = (x, y)
            self.grid.set(x, y, self.obstacles[i_obst])

        self.mission = "get to the green goal square"

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        dirs = [-1, 0, 1]
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            delta_x = random.choice(dirs)
            delta_y = random.choice(dirs)
            delta_pos = (delta_x, delta_y)
            new_pos = tuple(map(add, old_pos, delta_pos))

            while self.grid.get(*new_pos) != None and not self.grid.get(*new_pos).can_overlap():
                x_update = random.choice(dirs)
                y_update = random.choice(dirs)
                pos_update = (x_update, y_update)
                new_pos = tuple(map(add, old_pos, pos_update))

            self.grid.set(new_pos[0], new_pos[1], self.obstacles[i_obst])
            self.grid.set(old_pos[0], old_pos[1], None)
            self.obstacles[i_obst].cur_pos = new_pos
        return obs, reward, done, info


class DynamicObstaclesEnv5x5(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=5)


class DynamicObstaclesRandomEnv5x5(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)


class DynamicObstaclesEnv6x6(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=6)


class DynamicObstaclesRandomEnv6x6(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)


class DynamicObstaclesEnv16x16(DynamicObstaclesEnv):
    def __init__(self):
        super().__init__(size=16)


register(
    id='MiniGrid-Dynamic-Obstacles-5x5-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv5x5'
)

register(
    id='MiniGrid-Dynamic-Obstacles-Random-5x5-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesRandomEnv5x5'
)

register(
    id='MiniGrid-Dynamic-Obstacles-6x6-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv6x6'
)

register(
    id='MiniGrid-Dynamic-Obstacles-Random-6x6-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesRandomEnv6x6'
)

register(
    id='MiniGrid-Dynamic-Obstacles-8x8-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv'
)

register(
    id='MiniGrid-Dynamic-Obstacles-16x16-v0',
    entry_point='gym_minigrid.envs:DynamicObstaclesEnv16x16'
)