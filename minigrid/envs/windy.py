from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
from gymnasium.error import ResetNeeded

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


class WindyEnv(MiniGridEnv):
    """
    ## Description

    Reach a goal while a northward wind randomly displaces the agent. There is
    one wind probability per grid column, including the boundary columns. After
    each action, wind attempts to move the agent one cell north with the
    probability of the column occupied *before* the action. Wind never changes
    the agent's orientation and cannot push through a wall. Entering a goal
    ends the episode immediately, before any further movement.

    The default probabilities are 0.5 in columns 2 through width - 3 and zero
    elsewhere. Pass ``wind_probabilities`` to configure another profile; an
    all-zero profile gives a deterministic control environment. Wind uses the
    environment's seeded random generator, so reset(seed=...) reproduces it.
    Probabilities are environment parameters, not extra observation channels.

    With ``two_rooms=True``, a vertical wall divides the room, with a one-cell
    opening halfway down it. The start and goal are halfway down the left and
    right sides. This layout is inspired by the two-room windy gridworld in
    Distributional Reinforcement Learning with Quantile Regression
    (https://arxiv.org/abs/1710.10044). It is a Minigrid adaptation, not an exact
    reproduction: it retains egocentric observations, turning actions and the
    usual time-dependent reward, and uses Bernoulli one-cell wind rather than
    the paper's wind magnitudes and random-direction transitions.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name    | Action       |
    |-----|---------|--------------|
    | 0   | left    | Turn left    |
    | 1   | right   | Turn right   |
    | 2   | forward | Move forward |
    | 3   | pickup  | No operation |
    | 4   | drop    | No operation |
    | 5   | toggle  | No operation |
    | 6   | done    | No operation |

    All seven actions advance time and are followed by wind, including turns
    and no-ops. This environment has no objects to pick up, drop or toggle.

    ## Observation Encoding

    Standard Minigrid dictionary with ``image``, ``direction`` and ``mission``.
    Each image tile encodes (OBJECT_IDX, COLOR_IDX, STATE). The observation and
    rendered frame describe the state *after* both the action and wind.

    ## Rewards

    Success gives ``1 - 0.9 * (step_count / max_steps)``; other steps give zero.
    Wind is part of the same transition and does not consume another step.

    ## Termination

    Reaching the goal terminates the episode, including when pushed onto it.
    Reaching ``max_steps`` truncates the episode.

    ## Registered Configurations

    - ``MiniGrid-Windy-8x8-v0``
    - ``MiniGrid-Windy-TwoRooms-12x9-v0``
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        wind_probabilities: Sequence[float] | None = None,
        two_rooms: bool = False,
        max_steps: int | None = None,
        **kwargs,
    ):
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 5:
                raise ValueError(f"{name} must be an integer >= 5")
        width, height = int(width), int(height)
        if two_rooms and width < 7:
            raise ValueError("two_rooms requires width >= 7")
        if max_steps is None:
            max_steps = 4 * width * height
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, Integral)
            or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")

        if wind_probabilities is None:
            wind_probabilities = [
                0.5 if 2 <= x < width - 2 else 0.0 for x in range(width)
            ]
        if len(wind_probabilities) != width:
            raise ValueError(
                "wind_probabilities must contain one probability per column"
            )
        if any(
            not isinstance(p, Real) or not np.isfinite(p) or not 0 <= p <= 1
            for p in wind_probabilities
        ):
            raise ValueError("wind probabilities must be finite numbers in [0, 1]")
        # Own an immutable copy: caller mutation must not change the dynamics.
        self.wind_probabilities = tuple(float(p) for p in wind_probabilities)
        self.two_rooms = two_rooms
        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            width=width,
            height=height,
            max_steps=int(max_steps),
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        if self.two_rooms:
            self.grid.vert_wall(width // 2, 1, height - 2)
            self.grid.set(width // 2, height // 2, None)
        self.put_obj(Goal(), width - 2, height // 2)
        self.agent_pos = (1, height // 2)
        self.agent_dir = 0
        self.mission = self._gen_mission()

    def _move(self, position):
        """Apply one collision-checked displacement and its terminal reward."""
        x, y = (int(v) for v in position)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0, False
        cell = self.grid.get(x, y)
        if cell is not None and not cell.can_overlap():
            return 0.0, False
        self.agent_pos = (x, y)
        if cell is not None and cell.type == "goal":
            return self._reward(), True
        return 0.0, cell is not None and cell.type == "lava"

    def step(self, action):
        if self.agent_dir is None:
            raise ResetNeeded("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action: {action}")

        self.step_count += 1
        probability = self.wind_probabilities[self.agent_pos[0]]
        reward, terminated = 0.0, False
        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.forward:
            reward, terminated = self._move(self.front_pos)

        # Do not draw random numbers at p=0 or p=1, or after termination.
        if not terminated and probability > 0:
            if probability == 1 or self.np_random.random() < probability:
                reward, terminated = self._move(
                    (self.agent_pos[0], self.agent_pos[1] - 1)
                )

        truncated = self.step_count >= self.max_steps
        obs = self.gen_obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, {}
