from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class RedBlueDoorEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        self.size = size

        super().__init__(
            grid_size=2*size,
            max_steps=20*size*size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the grid walls
        self.grid.wall_rect(0, 0, 2*self.size, self.size)
        self.grid.wall_rect(self.size//2, 0, self.size, self.size)

        # Place the agent in the top-left corner
        self.place_agent(top=(self.size//2, 0), size=(self.size, self.size))

        # Add a red door at a random position in the left wall
        pos = self._rand_int(1, self.size - 1)
        self.red_door = Door("red")
        self.grid.set(self.size//2, pos, self.red_door)

        # Add a blue door at a random position in the right wall
        pos = self._rand_int(1, self.size - 1)
        self.blue_door = Door("blue")
        self.grid.set(self.size//2 + self.size - 1, pos, self.blue_door)

        # Generate the mission string
        self.mission = "open the red door then the blue door"

        # Set the resolution state to 0
        #   - 0 means "no door opened"
        #   - 1 means "red door opened"
        #   - 2 means "red then blue door opened"
        self.resolution_state = 0

    def step(self, action):
        red_door_opened_before = self.red_door.is_open
        blue_door_opened_before = self.blue_door.is_open

        obs, reward, done, info = MiniGridEnv.step(self, action)

        red_door_opened_after = self.red_door.is_open
        blue_door_opened_after = self.blue_door.is_open

        red_door_opened = red_door_opened_after and not(red_door_opened_before)
        blue_door_opened = blue_door_opened_after and not(blue_door_opened_before)

        if self.resolution_state == 0 and red_door_opened:
            self.resolution_state += 1
        elif self.resolution_state == 1 and blue_door_opened:
            self.resolution_state += 1
            reward = 1
            done = True

        return obs, reward, done, info

class RedBlueDoorEnv6x6(RedBlueDoorEnv):
    def __init__(self):
        super().__init__(size=6)

register(
    id='MiniGrid-RedBlueDoors-6x6-v0',
    entry_point='gym_minigrid.envs:RedBlueDoorEnv6x6'
)

register(
    id='MiniGrid-RedBlueDoors-8x8-v0',
    entry_point='gym_minigrid.envs:RedBlueDoorEnv'
)
