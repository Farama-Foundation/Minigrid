from __future__ import annotations

from operator import add

from gymnasium.spaces import Discrete

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Goal
from minigrid.minigrid_env import MiniGridEnv


class DynamicObstaclesEnv(MiniGridEnv):
    """
    ## Description

    This environment is an empty room with moving obstacles.
    The goal of the agent is to reach the green goal square without colliding
    with any obstacle. A large penalty is subtracted if the agent collides with
    an obstacle and the episode finishes. This environment is useful to test
    Dynamic Obstacle Avoidance for mobile robots with Reinforcement Learning in
    Partial Observability.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure. A '-1' penalty is
    subtracted if the agent collides with an obstacle.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent collides with an obstacle.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Dynamic-Obstacles-5x5-v0`
    - `MiniGrid-Dynamic-Obstacles-Random-5x5-v0`
    - `MiniGrid-Dynamic-Obstacles-6x6-v0`
    - `MiniGrid-Dynamic-Obstacles-Random-6x6-v0`
    - `MiniGrid-Dynamic-Obstacles-8x8-v0`
    - `MiniGrid-Dynamic-Obstacles-16x16-v0`

    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        n_obstacles=4,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Reduce obstacles if there are too many
        if n_obstacles <= size / 2 + 1:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size / 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        # Allow only 3 actions permitted: left, right, forward
        self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = "get to the green goal square"

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal"

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.place_obj(
                    self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100
                )
                self.grid.set(old_pos[0], old_pos[1], None)
            except Exception:
                pass

        # Update the agent's position/direction
        obs, reward, terminated, truncated, info = super().step(action)

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = -1
            terminated = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info
