from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal
import pdb  # debugging


class DynamicMiniGrid(MiniGridEnv):
    """
    DynamicMiniGrid: Mini Grid Environment, that can dynamically change, by altering a single tile
    """

    def __init__(self, size=8, agent_start_pos=(1, 1),agent_start_dir=0,):
        #Copied from EmptyEnv Todo: Make this class a child of EmptyEnv?
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__( grid_size=size, max_steps=4 * size * size,
                see_through_walls=False)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def alter(self):
        pdb.set_trace()

        def alter_start(start_position, reward_position):
            # if start position move it
            raise NotImplementedError

        def alter_reward_position(reward_position, start_position):
            raise NotImplementedError

        def alter_single_grid(current_value):
            raise NotImplementedError

        start_position = None # todo implement readout
        reward_position = None # todo implement readout

        while True:
            # pick random pile
            random_pile = 0


            if random_pile == start_position:
                alter_start(start_position)
            elif random_pile == reward_position:
                alter_reward_position(reward_position, start_position)
            else:
                a = 1

            # sanity check

            pass

