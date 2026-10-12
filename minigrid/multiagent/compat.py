"""Explicit list/tuple API adapter; not registered as a Gymnasium environment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from gymnasium import spaces


class MarlgridListAdapter:
    """Expose marlgrid-style joint calls around one Parallel environment.

    ``reset()`` returns a list; ``step(sequence)`` returns observations, reward
    array, global done, info. Slots remain in possible_agents order, including
    completed and delayed agents. Observation content is the wrapped environment's
    configured format, not pixel-identical legacy marlgrid rendering.

    The adapter never changes the wrapped environment's configuration, public
    method signatures, RNG sequence or global Gymnasium registry. Use its
    ``parallel_env`` attribute for access to the native Parallel interface.
    """

    def __init__(self, parallel_env):
        self.parallel_env = parallel_env
        self.possible_agents = tuple(parallel_env.possible_agents)
        self.action_space = spaces.Tuple(
            tuple(parallel_env.action_space(a) for a in self.possible_agents)
        )
        self.observation_space = spaces.Tuple(
            tuple(parallel_env.observation_space(a) for a in self.possible_agents)
        )
        self._pending_seed = None
        self._started = False

    def seed(self, seed=None):
        # Defer the seed to the next reset, never consume an extra reset here.
        self._pending_seed = seed
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        effective_seed = self._pending_seed if seed is None else seed
        observations, _ = self.parallel_env.reset(seed=effective_seed, options=options)
        self._pending_seed = None
        self._started = True
        return [observations[a] for a in self.possible_agents]

    def step(self, actions):
        if not self._started:
            raise RuntimeError("Call reset before step")
        if not isinstance(actions, (Sequence, np.ndarray)) or isinstance(
            actions, (str, bytes)
        ):
            raise ValueError("Provide a sequence in possible_agents order")
        if len(actions) != len(self.possible_agents):
            raise ValueError("Provide one action slot for every possible agent")
        # Validate even ignored slots, before any mutation of the environment.
        if not all(
            self.parallel_env.action_space(a).contains(value)
            for a, value in zip(self.possible_agents, actions)
        ):
            raise ValueError("Invalid action in joint sequence")
        env = self.parallel_env
        active_actions = {
            a: value
            for a, value in zip(self.possible_agents, actions)
            if a in env.agents
        }
        _, rewards, terminations, truncations, infos = env.step(active_actions)
        observations = [env.observe(a) for a in self.possible_agents]
        reward_array = np.asarray(
            [rewards.get(a, 0.0) for a in self.possible_agents], dtype=np.float64
        )
        info = {
            "terminations": terminations,
            "truncations": truncations,
            "agents": infos,
        }
        return observations, reward_array, not env.agents, info

    def render(self):
        return self.parallel_env.render()

    def close(self):
        self.parallel_env.close()


def make_marlgrid(env_id, **kwargs):
    """Build a legacy named preset without modifying Gymnasium's registry.

    Names and sizes follow the source, including the historic 1AgentCluttered15x15
    name that configures an 11x11 grid. Observations default to RGB; calls use the
    list adapter's four-return-value API, not Gymnasium's single-agent API.
    """
    from minigrid.multiagent.env import ClutteredParallelEnv, EmptyParallelEnv
    from minigrid.multiagent.tasks import GoalCycleParallelEnv

    presets = {
        "MarlGrid-1AgentCluttered15x15-v0": (
            ClutteredParallelEnv,
            dict(size=11, num_agents=1, n_clutter=30, agent_view_size=5),
        ),
        "MarlGrid-3AgentCluttered11x11-v0": (
            ClutteredParallelEnv,
            dict(size=11, num_agents=3, clutter_density=0.15),
        ),
        "MarlGrid-3AgentCluttered15x15-v0": (
            ClutteredParallelEnv,
            dict(size=15, num_agents=3, clutter_density=0.15),
        ),
        "MarlGrid-2AgentEmpty9x9-v0": (EmptyParallelEnv, dict(size=9, num_agents=2)),
        "MarlGrid-3AgentEmpty9x9-v0": (EmptyParallelEnv, dict(size=9, num_agents=3)),
        "MarlGrid-4AgentEmpty9x9-v0": (EmptyParallelEnv, dict(size=9, num_agents=4)),
        "Goalcycle-demo-solo-v0": (
            GoalCycleParallelEnv,
            dict(
                size=13,
                num_agents=1,
                clutter_density=0.1,
                n_bonus_tiles=3,
                view_offset=1,
            ),
        ),
    }
    if env_id not in presets:
        raise ValueError(f"Unknown marlgrid preset: {env_id}")
    factory, defaults = presets[env_id]
    if "n_clutter" in kwargs:
        defaults.pop("clutter_density", None)
    if "clutter_density" in kwargs:
        defaults.pop("n_clutter", None)
    return MarlgridListAdapter(
        factory(
            **{"observation_style": "image", "view_tile_size": 8, **defaults, **kwargs}
        )
    )
