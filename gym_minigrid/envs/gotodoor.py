from gym_minigrid.minigrid import COLOR_NAMES, Door, Grid, MiniGridEnv, MissionSpace


class GoToDoorEnv(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(self, size=5, **kwargs):
        assert size >= 5
        self.size = size
        mission_space = MissionSpace(
            mission_func=lambda color: f"go to the {color} door",
            ordered_placeholders=[COLOR_NAMES],
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
        # Create the grid
        self.grid = Grid(width, height)

        # Randomly vary the room width and height
        width = self._rand_int(5, width + 1)
        height = self._rand_int(5, height + 1)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the 4 doors at random positions
        doorPos = []
        doorPos.append((self._rand_int(2, width - 2), 0))
        doorPos.append((self._rand_int(2, width - 2), height - 1))
        doorPos.append((0, self._rand_int(2, height - 2)))
        doorPos.append((width - 1, self._rand_int(2, height - 2)))

        # Generate the door colors
        doorColors = []
        while len(doorColors) < len(doorPos):
            color = self._rand_elem(COLOR_NAMES)
            if color in doorColors:
                continue
            doorColors.append(color)

        # Place the doors in the grid
        for idx, pos in enumerate(doorPos):
            color = doorColors[idx]
            self.grid.set(*pos, Door(color))

        # Randomize the agent start position and orientation
        self.place_agent(size=(width, height))

        # Select a random target door
        doorIdx = self._rand_int(0, len(doorPos))
        self.target_pos = doorPos[doorIdx]
        self.target_color = doorColors[doorIdx]

        # Generate the mission string
        self.mission = "go to the %s door" % self.target_color

    def step(self, action):
        obs, reward, done, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Don't let the agent open any of the doors
        if action == self.actions.toggle:
            done = True

        # Reward performing done action in front of the target door
        if action == self.actions.done:
            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                reward = self._reward()
            done = True

        return obs, reward, done, info
