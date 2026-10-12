"""Multi-agent navigation using current MiniGrid objects and a Parallel API.

Actions are submitted together, then resolved in a seeded random order. Like
marlgrid, ghost mode defaults to True and permits overlapping agents. Goal and
lava completion is individual; the remaining agents continue playing.

The original single-agent environments and registry are not modified.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from minigrid.core.actions import Actions
from minigrid.core.constants import (
    COLOR_NAMES,
    COLOR_TO_IDX,
    COLORS,
    DIR_TO_VEC,
    OBJECT_TO_IDX,
)
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.multiagent.observations import ObservationSettings
from minigrid.utils.rendering import fill_coords, point_in_triangle, rotate_fn

try:
    from pettingzoo import ParallelEnv
except ModuleNotFoundError as error:
    if error.name != "pettingzoo":
        raise
    # Importing Minigrid or collecting its doctests must not require this extra.
    ParallelEnv = object


@dataclass
class AgentState:
    position: tuple[int, int] | None
    direction: int
    color: str
    carrying: WorldObj | None = None
    prestige: float = 0.0
    bonus_state: int | None = None
    bonuses: list[tuple[int, float]] = field(default_factory=list)
    last_bonus: int | None = None


class _AgentMarker(WorldObj):
    """Observation-only marker; never replaces the underlying world object."""

    def __init__(self, color, direction):
        super().__init__("agent", color)
        self.direction = direction

    def encode(self):
        return OBJECT_TO_IDX["agent"], COLOR_TO_IDX[self.color], self.direction

    def render(self, image):
        triangle = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
        triangle = rotate_fn(triangle, 0.5, 0.5, 0.5 * math.pi * self.direction)
        fill_coords(image, triangle, COLORS[self.color])


class MultiGridParallelEnv(ParallelEnv):
    """Shared grid, per-agent egocentric observations and individual rewards.

    ``reset(seed=...)`` returns observations and infos keyed by agent ID.
    ``step(actions)`` requires exactly one action for every live agent and returns
    the five PettingZoo dictionaries. Terminal observations are returned once;
    dead agents must be omitted from the next action dictionary.

    The seven actions and observation fields match Minigrid. Other agents use
    the existing ``agent`` object code, their color, and direction relative to
    the observing agent's view. ``render_mode='rgb_array'`` returns a global
    frame. Single-agent Gymnasium wrappers are not adapters for this API.

    Default navigation rules follow marlgrid: agents may overlap, starts need
    not be reachable, the horizon is 100 steps, and goal rewards decay with
    time. ``ensure_reachable=True`` changes the start/map distribution and
    excludes goal starts. Unlike legacy marlgrid, every reset restores heading
    zero, rather than preserving each agent's previous heading.
    """

    metadata = {
        "name": "minigrid_multiagent_v0",
        "render_modes": ["rgb_array"],
        "render_fps": 10,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        size: int = 9,
        num_agents: int = 2,
        n_clutter: int = 0,
        max_steps: int | None = None,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        ghost_mode: bool = True,
        reward_decay: bool = True,
        randomize_goal: bool = False,
        ensure_reachable: bool = False,
        respawn: bool = False,
        spawn_delay=0,
        prestige_beta: float = 0.95,
        allow_negative_prestige: bool = False,
        observation_style: str = "symbolic",
        view_tile_size: int = 5,
        view_offset: int = 0,
        see_through_walls: bool = False,
        observe_rewards: bool = False,
        observe_position: bool = False,
        observe_orientation: bool = False,
        hide_item_types=(),
        restrict_actions: bool = False,
    ):
        if ParallelEnv is object:
            raise ImportError(
                "Multi-agent environments require pip install 'minigrid[multiagent]'"
            )
        for name, value, minimum in (
            ("size", size, 5),
            ("num_agents", num_agents, 1),
            ("n_clutter", n_clutter, 0),
            ("agent_view_size", agent_view_size, 3),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if num_agents > len(COLOR_NAMES):
            raise ValueError(
                f"At most {len(COLOR_NAMES)} distinct agent colors are supported"
            )
        if agent_view_size % 2 != 1:
            raise ValueError("agent_view_size must be odd")
        for name, value in (
            ("ghost_mode", ghost_mode),
            ("reward_decay", reward_decay),
            ("randomize_goal", randomize_goal),
            ("ensure_reachable", ensure_reachable),
            ("respawn", respawn),
            ("allow_negative_prestige", allow_negative_prestige),
            ("restrict_actions", restrict_actions),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        required_cells = max(1, num_agents if not ghost_mode else 1)
        if ensure_reachable:
            required_cells += 1  # This opt-in mode excludes goal-cell starts.
        if n_clutter + required_cells > (size - 2) ** 2:
            raise ValueError("Not enough cells for the requested placement mode")
        if max_steps is None:
            max_steps = 100
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, Integral)
            or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only render_mode=None or 'rgb_array' is supported")

        self.size = int(size)
        self.n_clutter = int(n_clutter)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.ghost_mode = ghost_mode
        self.reward_decay = reward_decay
        self.randomize_goal = randomize_goal
        self.ensure_reachable = ensure_reachable
        self._configure_lifecycle(
            spawn_delay, num_agents, respawn, prestige_beta, allow_negative_prestige
        )
        self.goal_pos = None
        self.actions = Actions
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = []
        self._visible_agents = []
        self.agent_states = {}
        self.grid = None
        self.step_count = 0
        self._spawn_region = None
        self._last_rewards = {}
        self.np_random, _ = seeding.np_random(None)
        self._view = MiniGridEnv(
            mission_space=MissionSpace(mission_func=self._mission),
            grid_size=self.size,
            agent_view_size=int(agent_view_size),
            max_steps=self.max_steps,
        )
        self.observation_settings = ObservationSettings(
            observation_style,
            int(agent_view_size),
            view_tile_size,
            view_offset,
            see_through_walls,
            observe_rewards,
            observe_position,
            observe_orientation,
            (
                tuple(hide_item_types)
                if not isinstance(hide_item_types, str)
                else hide_item_types
            ),
        )
        self.observation_spaces = {
            agent: self.observation_settings.space(self._view.observation_space)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(3 if restrict_actions else 7)
            for agent in self.possible_agents
        }

    def _configure_lifecycle(self, delays, count, respawn, beta, negative):
        if isinstance(delays, Integral) and not isinstance(delays, bool):
            delays = [delays] * count
        if (
            not isinstance(delays, (list, tuple))
            or len(delays) != count
            or any(
                isinstance(d, bool) or not isinstance(d, Integral) or d < 0
                for d in delays
            )
        ):
            raise ValueError(
                "spawn_delay must be a nonnegative integer or one per agent"
            )
        if (
            isinstance(beta, bool)
            or not isinstance(beta, Real)
            or not np.isfinite(beta)
            or not 0 <= beta <= 1
        ):
            raise ValueError("prestige_beta must be finite and in [0, 1]")
        self.spawn_delays = dict(
            zip((f"agent_{i}" for i in range(count)), map(int, delays))
        )
        self.respawn = respawn
        self.prestige_beta = float(beta)
        self.allow_negative_prestige = negative

    @staticmethod
    def _mission():
        return "get to the green goal square"

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _reachable(self):
        goal = self.goal_pos
        seen = {goal}
        frontier = deque([goal])
        while frontier:
            x, y = frontier.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                pos = (x + dx, y + dy)
                cell = self.grid.get(*pos)
                if pos not in seen and (cell is None or cell.can_overlap()):
                    seen.add(pos)
                    frontier.append(pos)
        return seen

    def _sample_position(self, accepts, max_tries=100000):
        # Match marlgrid's rejection sampling over the whole grid, including
        # rejected boundary cells. Do not silently change the map distribution.
        for _ in range(max_tries):
            position = tuple(
                int(v) for v in self.np_random.integers(0, self.size, size=2)
            )
            if accepts(position):
                return position
        raise RuntimeError(f"Placement failed after {max_tries} attempts")

    def _generate_grid(self):
        for _ in range(100 if self.ensure_reachable else 1):
            self.grid = Grid(self.size, self.size)
            self.grid.wall_rect(0, 0, self.size, self.size)
            self.goal_pos = (
                self._sample_position(lambda p: self.grid.get(*p) is None, 100)
                if self.randomize_goal
                else (self.size - 2, self.size - 2)
            )
            self.grid.set(*self.goal_pos, Goal())
            for _ in range(self.n_clutter):
                pos = self._sample_position(lambda p: self.grid.get(*p) is None, 100)
                self.grid.set(*pos, Wall())
            if not self.ensure_reachable:
                return None
            reachable = self._reachable() - {self.goal_pos}
            required = 1 if self.ghost_mode else len(self.possible_agents)
            if len(reachable) >= required:
                return reachable
        raise RuntimeError(
            "Could not generate enough reachable start cells in 100 attempts"
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
            for i, agent in enumerate(self.possible_agents):
                self.action_spaces[agent].seed(seed + i)
                self.observation_spaces[agent].seed(seed + i)
        self.agents = []
        self.agent_states = {}
        self._visible_agents = []
        self.step_count = 0
        self._last_rewards = {agent: 0.0 for agent in self.possible_agents}
        self._spawn_region = self._generate_grid()
        for i, agent in enumerate(self.possible_agents):
            self.agent_states[agent] = AgentState(None, 0, COLOR_NAMES[i])
            if self.spawn_delays[agent] == 0:
                self._spawn_agent(agent)
        self.agents = self.possible_agents[:]
        self._visible_agents = [
            a for a in self.agents if self.agent_states[a].position is not None
        ]
        return {agent: self.observe(agent) for agent in self.agents}, {
            agent: {"active": agent in self._visible_agents} for agent in self.agents
        }

    def _spawn_agent(self, agent):
        def accepts(pos):
            cell = self.grid.get(*pos)
            occupied = any(
                s.position == pos for a, s in self.agent_states.items() if a != agent
            )
            return (
                (cell is None or cell.can_overlap())
                and (self.ghost_mode or not occupied)
                and (self._spawn_region is None or pos in self._spawn_region)
            )

        self.agent_states[agent].position = self._sample_position(accepts)
        self.agent_states[agent].carrying = None

    def _set_view(self, agent, grid):
        state = self.agent_states[agent]
        self._view.grid = grid
        self._view.agent_pos = state.position
        self._view.agent_dir = state.direction
        self._view.carrying = state.carrying
        self._view.mission = self._mission()
        return self._view

    def _observation_grid(self, observer=None):
        grid = self.grid.copy()
        # Keep a stable representative when ghost-mode agents share a cell.
        for agent in reversed(self._visible_agents):
            state = self.agent_states[agent]
            direction = state.direction
            if observer is not None:
                direction = (direction - self.agent_states[observer].direction + 3) % 4
            grid.set(*state.position, _AgentMarker(state.color, direction))
        return grid

    def observe(self, agent):
        if agent not in self.agent_states:
            raise ValueError("Call reset and use an ID from possible_agents")
        return self.observation_settings.observe(self, agent)

    def _enter_cell(self, agent, cell):
        if cell is None or cell.type not in ("goal", "lava"):
            return 0.0, False
        return (1.0 if cell.type == "goal" else 0.0), True

    def _after_action(self, agent, reward):
        state = self.agent_states[agent]
        if self.allow_negative_prestige or reward >= 0:
            state.prestige += reward
        else:
            state.prestige = 0.0
        if state.last_bonus is not None:
            state.bonuses.append((state.last_bonus, state.prestige))
        state.prestige *= self.prestige_beta

    def _apply_action(self, agent, action, occupied):
        state = self.agent_states[agent]
        state.last_bonus = None
        if action == Actions.left:
            state.direction = (state.direction - 1) % 4
        elif action == Actions.right:
            state.direction = (state.direction + 1) % 4
        elif action != Actions.done:
            dx, dy = DIR_TO_VEC[state.direction]
            front = (state.position[0] + int(dx), state.position[1] + int(dy))
            cell = self.grid.get(*front)
            peers = occupied.get(front, [])
            if cell is None and peers:
                # In marlgrid the top-level cell is an agent on an otherwise
                # empty tile. Passable objects (e.g. goals) can hold agents even
                # with ghost_mode=False; preserve that distinction.
                if action != Actions.forward or not self.ghost_mode:
                    return 0.0, False
            if action == Actions.forward and (cell is None or cell.can_overlap()):
                occupied[state.position].remove(agent)
                state.position = front
                occupied.setdefault(front, []).append(agent)
                return self._enter_cell(agent, cell)
            elif (
                action == Actions.pickup
                and cell is not None
                and cell.can_pickup()
                and state.carrying is None
            ):
                state.carrying = cell
                cell.cur_pos = np.array([-1, -1])
                self.grid.set(*front, None)
            elif action == Actions.drop and cell is None and state.carrying is not None:
                self.grid.set(*front, state.carrying)
                state.carrying.cur_pos = front
                state.carrying = None
            elif action == Actions.toggle and cell is not None:
                cell.toggle(self._set_view(agent, self.grid), front)
        return 0.0, False

    def step(self, actions):
        if self.grid is None:
            raise RuntimeError("Call reset before step")
        if set(actions) != set(self.agents):
            raise ValueError("Provide exactly one action for each live agent")
        if not self.agents:
            return {}, {}, {}, {}, {}
        for agent, action in actions.items():
            if not self.action_spaces[agent].contains(action):
                raise ValueError(f"Invalid action for {agent}: {action}")

        acting_agents = self.agents[:]
        # Waiting identities remain live Parallel API agents, but their actions
        # are ignored until activation (before the shared step is incremented).
        for agent in acting_agents:
            if (
                self.agent_states[agent].position is None
                and self.step_count >= self.spawn_delays[agent]
            ):
                self._spawn_agent(agent)
        occupied = {}
        for agent in acting_agents:
            position = self.agent_states[agent].position
            if position is not None:
                occupied.setdefault(position, []).append(agent)
        self.step_count += 1
        rewards = {a: 0.0 for a in acting_agents}
        terminated = {a: False for a in acting_agents}
        # The source shuffles all agent identities, including inactive ones.
        # Keep the same scheduling contract after the Parallel API removes an ID.
        for index in self.np_random.permutation(len(self.possible_agents)):
            agent = self.possible_agents[int(index)]
            if agent not in actions or self.agent_states[agent].position is None:
                continue
            rewards[agent], terminated[agent] = self._apply_action(
                agent, actions[agent], occupied
            )
            if self.reward_decay:
                rewards[agent] *= 1 - 0.9 * self.step_count / self.max_steps
            self._after_action(agent, rewards[agent])
        infos = {a: {"respawned": False} for a in acting_agents}
        for agent in acting_agents:
            if terminated[agent] and self.respawn:
                # A respawn is an in-episode transition, not termination followed
                # by illegal resurrection of a dead PettingZoo agent identity.
                self._spawn_agent(agent)
                terminated[agent] = False
                infos[agent]["respawned"] = True
        truncated = {
            a: self.step_count >= self.max_steps and not terminated[a]
            for a in acting_agents
        }
        # Completed agents disappear before the next observations are generated.
        # Timed-out agents remain visible in the terminal observation and frame.
        self._visible_agents = [
            a
            for a in acting_agents
            if not terminated[a] and self.agent_states[a].position is not None
        ]
        self.agents = [
            a for a in acting_agents if not terminated[a] and not truncated[a]
        ]
        self._last_rewards = rewards.copy()
        observations = {a: self.observe(a) for a in acting_agents}
        for agent in acting_agents:
            infos[agent]["active"] = agent in self._visible_agents
            infos[agent]["prestige"] = self.agent_states[agent].prestige
        return observations, rewards, terminated, truncated, infos

    def render(self):
        if self.render_mode is None:
            return None
        if self.grid is None:
            raise RuntimeError("Call reset before render")
        return self._observation_grid().render(tile_size=32, agent_pos=(-1, -1))

    def close(self):
        self._view.close()


class EmptyParallelEnv(MultiGridParallelEnv):
    """Marlgrid-style empty-room navigation with two agents by default."""

    def __init__(self, size=9, num_agents=2, **kwargs):
        super().__init__(size=size, num_agents=num_agents, n_clutter=0, **kwargs)


class ClutteredParallelEnv(MultiGridParallelEnv):
    """Marlgrid navigation with wall clutter and an optional randomized goal.

    Specify either ``n_clutter`` or ``clutter_density``. Density is multiplied
    by the interior area and rounded down. If neither is supplied, use 0.15,
    matching the source's registered presets (its raw class requires a value).
    """

    def __init__(
        self, size=11, num_agents=3, n_clutter=None, clutter_density=None, **kwargs
    ):
        if n_clutter is not None and clutter_density is not None:
            raise ValueError("Provide n_clutter or clutter_density, not both")
        if n_clutter is None and clutter_density is None:
            clutter_density = 0.15  # marlgrid's registered 3-agent presets
        if clutter_density is not None:
            if (
                isinstance(clutter_density, bool)
                or not isinstance(clutter_density, Real)
                or not np.isfinite(clutter_density)
                or not 0 <= clutter_density <= 1
            ):
                raise ValueError("clutter_density must be finite and in [0, 1]")
            if isinstance(size, bool) or not isinstance(size, Integral) or size < 5:
                raise ValueError("size must be an integer >= 5")
            n_clutter = int(clutter_density * (size - 2) ** 2)
        super().__init__(
            size=size, num_agents=num_agents, n_clutter=n_clutter, **kwargs
        )
