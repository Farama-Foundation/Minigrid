from gym_minigrid.minigrid import Goal, Grid, Lava, MiniGridEnv, MissionSpace


class DistShiftEnv(MiniGridEnv):
    """
    Distributional shift environment.
    """

    def __init__(
        self,
        width=9,
        height=7,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        strip2_row=2,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = (width - 2, 1)
        self.strip2_row = strip2_row

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place the lava rows
        for i in range(self.width - 6):
            self.grid.set(3 + i, 1, Lava())
            self.grid.set(3 + i, self.strip2_row, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"
