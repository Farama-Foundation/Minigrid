from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Lava, Wall


class DynamicMiniGrid(MiniGridEnv):
    """
    DynamicMiniGrid: Mini Grid Environment, that can dynamically change, by altering a single tile
    """

    def __init__(self, size=8, agent_start_pos=(1, 1),agent_start_dir=0, agent_view_size=7):

        # Copied from EmptyEnv Todo: Make this class a child of EmptyEnv?
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__( grid_size=size, max_steps=4 * size * size,
                          see_through_walls=False, agent_view_size=agent_view_size)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner # Todo - should this change?
        self.goal_pos = (width-2, height-2)
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def alter(self):
    # Todo: alternatively use place_obj function (pre-selection of obj)
    # Todo: add int for multiple repetion

        def alter_start():
            # Todo: use self.place_agent()
            while True:
                new_pos = (self.np_random.randint(1,self.height-1),
                           self.np_random.randint(1,self.width-1))
                if new_pos != self.goal_pos and new_pos != self.agent_start_pos:
                    self.agent_start_pos = new_pos
                    return

        def alter_goal_position():
            while True:
                new_pos = (self.np_random.randint(1,self.height-1),
                           self.np_random.randint(1,self.width-1))
                if new_pos != self.goal_pos and new_pos != self.agent_start_pos:
                    self.put_obj(Goal(), *new_pos)
                    self.grid.set(*self.goal_pos, None)
                    return

        def alter_single_grid(random_pile):
            possible_obj = (Lava(), Wall(), None)
            prob_obj = (0.2, 0.2, 0.6)  # Todo: make this dependent on a constructor variable difficulty
            random_pile_idx = random_pile[0]*self.width + random_pile[1]
            # Ugly way of getting the right index in self.grid.grid Todo: improve
            while True:
                new_obj = self.np_random.choice(possible_obj, p=prob_obj) # todo use array of possibilities
                # can't do type checking on None
                if new_obj == None:
                    if self.grid.grid[random_pile_idx] != None:
                        self.grid.grid[random_pile_idx] = None
                        return
                elif self.grid.grid[random_pile_idx] == None:
                    self.put_obj(new_obj, *random_pile)
                    return
                elif new_obj.type != self.grid.grid[random_pile_idx].type:
                    return

        def grid_is_solvable(self):
            return True
            #raise NotImplementedError

        while True:
            # pick random tile
            random_tile = (self.np_random.randint(1,self.height-1),
                           self.np_random.randint(1,self.width-1)) # no manip of boarder -> 1, max-1
            if random_tile == self.agent_start_pos:
                alter_start()
            elif random_tile == self.goal_pos:
                alter_goal_position()
            else:
                alter_single_grid(random_tile)
            breakpoint()
            # sanity check
            if grid_is_solvable(self):
                return
            else:
                # reject change and do another way
                raise NotImplementedError

# Tests: 1000x alter -