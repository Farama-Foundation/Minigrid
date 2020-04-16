from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class DoorKeyOptionalEnv(MiniGridEnv):
    """
    Environment with a yellow door and no key, sparse reward

    Agent cannot solve task unless it is already carrying a yellow key 
    when the environment is initialized.
    """

    def __init__(self, size=8, key_color=None):
        self._key_color = key_color  # must be yellow to solve task        
        super().__init__(
            grid_size=size,
            max_steps=10*size*size
        )

    def reset(self):
        """Override reset so that agent can be initialized
        carrying a key already"""

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried
        if self._key_color is not None:
            self.carrying = Key(color=self._key_color)
        else:
            self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        self.mission = "use the key to open the door and then get to the goal"


class DoorHasKey8x8Env(DoorKeyOptionalEnv):
    def __init__(self):
        super().__init__(size=8, key_color='yellow')


class DoorNoKey8x8Env(DoorKeyOptionalEnv):
    def __init__(self):
        super().__init__(size=8, key_color=None)


register(
    id='MiniGrid-DoorHasKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorHasKey8x8Env'
)

register(
    id='MiniGrid-DoorNoKey-8x8-v0',
    entry_point='gym_minigrid.envs:DoorNoKey8x8Env'
)
