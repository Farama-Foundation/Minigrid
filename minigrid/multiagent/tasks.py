"""Additional marlgrid task families without changing the single-agent registry.

Behavioral reference: kandouss/marlgrid revision e88c40b, envs/doorkey.py,
envs/goalcycle.py, envs/viz_test.py and objects.BonusTile. Implemented using
current MiniGrid objects; see docs/content/multiagent.md for migration differences.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Floor, Goal, Key, Wall
from minigrid.multiagent.env import ClutteredParallelEnv, MultiGridParallelEnv


class DoorKeyParallelEnv(MultiGridParallelEnv):
    """A shared key and locked door separate two rooms from the common goal.

    Like marlgrid, starts are sampled on either side of the dividing wall.
    ``start_in_left_room=True`` explicitly selects the harder key/door variant.
    """

    def __init__(self, size=9, num_agents=2, start_in_left_room=False, **kwargs):
        if not isinstance(start_in_left_room, bool):
            raise ValueError("start_in_left_room must be a boolean")
        if kwargs.get("ensure_reachable", False) or kwargs.get("randomize_goal", False):
            raise ValueError(
                "DoorKey does not support ensure_reachable or randomize_goal"
            )
        self.start_in_left_room = start_in_left_room
        self.split_index = self.door_pos = None
        super().__init__(size=size, num_agents=num_agents, n_clutter=0, **kwargs)

    @staticmethod
    def _mission():
        return "use the key to open the door and then get to the goal"

    def _generate_grid(self):
        self.grid = Grid(self.size, self.size)
        self.grid.wall_rect(0, 0, self.size, self.size)
        self.goal_pos = (self.size - 2, self.size - 2)
        self.grid.set(*self.goal_pos, Goal())
        self.split_index = int(self.np_random.integers(2, self.size - 2))
        self.grid.vert_wall(self.split_index, 0)
        self.door_pos = (
            self.split_index,
            int(self.np_random.integers(1, self.size - 2)),
        )
        self.grid.set(*self.door_pos, Door("yellow", is_locked=True))
        for _ in range(100000):
            pos = tuple(
                int(v)
                for v in self.np_random.integers((0, 0), (self.split_index, self.size))
            )
            if self.grid.get(*pos) is None:
                self.grid.set(*pos, Key("yellow"))
                break
        else:
            raise RuntimeError("Key placement failed after 100000 attempts")
        if self.start_in_left_room:
            region = {
                (x, y)
                for x in range(1, self.split_index)
                for y in range(1, self.size - 1)
                if self.grid.get(x, y) is None
            }
            if not self.ghost_mode and len(region) < len(self.possible_agents):
                raise RuntimeError("Left room is too small for non-overlapping starts")
            return region
        return None


class _BonusTile(Floor):
    """A local floor subtype; no additions to global object encodings or caches."""

    def __init__(self, bonus_id):
        super().__init__("yellow")
        self.bonus_id = bonus_id

    def encode(self):
        kind, color, _ = super().encode()
        return kind, color, self.bonus_id


class GoalCycleParallelEnv(ClutteredParallelEnv):
    """Visit bonus tiles in cyclic order, independently for each agent.

    Entering a bonus tile never terminates the agent. A repeated or out-of-order
    visit receives ``-abs(penalty)``; first visits can be unscored. The cycle
    state is preserved across in-episode respawns and cleared on reset.
    """

    def __init__(
        self,
        size=13,
        num_agents=2,
        n_bonus_tiles=3,
        reward=1.0,
        penalty=0.0,
        initial_reward=True,
        reset_on_mistake=False,
        n_clutter=None,
        clutter_density=None,
        **kwargs,
    ):
        if (
            isinstance(n_bonus_tiles, bool)
            or not isinstance(n_bonus_tiles, Integral)
            or not 1 <= n_bonus_tiles <= 256
        ):
            raise ValueError("n_bonus_tiles must be an integer in [1, 256]")
        for name, value in (("reward", reward), ("penalty", penalty)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not isinstance(initial_reward, bool) or not isinstance(
            reset_on_mistake, bool
        ):
            raise ValueError("initial_reward and reset_on_mistake must be booleans")
        if kwargs.get("ensure_reachable", False) or kwargs.get("randomize_goal", False):
            raise ValueError(
                "GoalCycle does not support ensure_reachable or randomize_goal"
            )
        if n_clutter is None and clutter_density is None:
            clutter_density = 0.1
        kwargs.setdefault("reward_decay", False)
        self.n_bonus_tiles = int(n_bonus_tiles)
        self.reward = float(reward)
        self.penalty = abs(float(penalty))
        self.initial_reward = initial_reward
        self.reset_on_mistake = reset_on_mistake
        self.bonus_positions = []
        super().__init__(
            size=size,
            num_agents=num_agents,
            n_clutter=n_clutter,
            clutter_density=clutter_density,
            **kwargs,
        )
        if self.n_clutter + self.n_bonus_tiles > (self.size - 2) ** 2:
            raise ValueError("Not enough cells for bonus tiles and clutter")

    @staticmethod
    def _mission():
        return "cycle between yellow goal tiles"

    def _generate_grid(self):
        self.grid = Grid(self.size, self.size)
        self.grid.wall_rect(0, 0, self.size, self.size)
        self.goal_pos = None
        self.bonus_positions = []
        for index in range(self.n_bonus_tiles):
            pos = self._sample_position(lambda p: self.grid.get(*p) is None, 100)
            self.grid.set(*pos, _BonusTile(index))
            self.bonus_positions.append(pos)
        for _ in range(self.n_clutter):
            pos = self._sample_position(lambda p: self.grid.get(*p) is None, 100)
            self.grid.set(*pos, Wall())
        return None

    def _enter_cell(self, agent, cell):
        if not isinstance(cell, _BonusTile):
            return super()._enter_cell(agent, cell)
        state = self.agent_states[agent]
        first = state.bonus_state is None
        previous = (
            (cell.bonus_id - 1) % self.n_bonus_tiles if first else state.bonus_state
        )
        correct = (
            previous != cell.bonus_id
            and (previous + 1) % self.n_bonus_tiles == cell.bonus_id
        )
        state.bonus_state = (
            cell.bonus_id if correct or self.reset_on_mistake else previous
        )
        state.last_bonus = cell.bonus_id
        reward = self.reward if correct else -self.penalty
        return (0.0 if first and not self.initial_reward else reward), False


class VisibilityParallelEnv(MultiGridParallelEnv):
    """Wall-occlusion diagnostic from marlgrid; episodes end only at the horizon."""

    def __init__(self, size=9, num_agents=2, **kwargs):
        if kwargs.get("ensure_reachable", False) or kwargs.get("randomize_goal", False):
            raise ValueError(
                "Visibility does not support ensure_reachable or randomize_goal"
            )
        super().__init__(size=size, num_agents=num_agents, n_clutter=0, **kwargs)

    @staticmethod
    def _mission():
        return "observe the other agents around the wall"

    def _generate_grid(self):
        self.grid = Grid(self.size, self.size)
        self.grid.wall_rect(0, 0, self.size, self.size)
        self.grid.horz_wall(0, self.size // 2, self.size - 3)
        self.goal_pos = None
        return None
