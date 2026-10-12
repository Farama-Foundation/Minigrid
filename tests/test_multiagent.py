from __future__ import annotations

import copy
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import data_equivalence

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.world_object import Ball, Door, Floor, Key, Lava, Wall
from minigrid.multiagent import ClutteredParallelEnv, EmptyParallelEnv

pz_tests = pytest.importorskip("pettingzoo.test")


@pytest.mark.parametrize("env_class", [EmptyParallelEnv, ClutteredParallelEnv])
def test_parallel_api(env_class):
    env = env_class(max_steps=25)
    try:
        pz_tests.parallel_api_test(env, num_cycles=100)
    finally:
        env.close()


@pytest.mark.parametrize("env_class", [EmptyParallelEnv, ClutteredParallelEnv])
def test_parallel_seed(env_class):
    pz_tests.parallel_seed_test(lambda: env_class(max_steps=25), num_cycles=100)


@pytest.mark.parametrize("env_class", [EmptyParallelEnv, ClutteredParallelEnv])
def test_parallel_to_aec_adapter(env_class):
    from pettingzoo.utils import parallel_to_aec

    env = parallel_to_aec(env_class(max_steps=25))
    try:
        pz_tests.api_test(env, num_cycles=100)
    finally:
        env.close()


def test_action_dictionary_order_does_not_change_random_priority():
    first = placed_env()
    second = pickle.loads(pickle.dumps(first))
    for _ in range(20):
        actions = {agent: 2 for agent in first.agents}
        reversed_actions = dict(reversed(list(actions.items())))
        assert data_equivalence(first.step(actions), second.step(reversed_actions))
        assert first.agent_states == second.agent_states


def test_step_and_render_require_reset():
    env = EmptyParallelEnv(render_mode="rgb_array")
    with pytest.raises(RuntimeError, match="reset"):
        env.step({})
    with pytest.raises(RuntimeError, match="reset"):
        env.render()


def test_clutter_generation_is_bounded(monkeypatch):
    env = ClutteredParallelEnv(ensure_reachable=True)
    calls = []

    def unreachable():
        calls.append(True)
        return set()

    monkeypatch.setattr(env, "_reachable", unreachable)
    with pytest.raises(RuntimeError, match="100 attempts"):
        env.reset(seed=0)
    assert len(calls) == 100
    assert env.agents == env._visible_agents == []


def placed_env(**kwargs):
    kwargs.setdefault("ghost_mode", False)
    env = EmptyParallelEnv(**kwargs)
    env.reset(seed=42)
    env.agent_states["agent_0"].position = (2, 3)
    env.agent_states["agent_0"].direction = 0
    env.agent_states["agent_1"].position = (4, 3)
    env.agent_states["agent_1"].direction = 2
    return env


def test_collision_has_one_winner_and_no_overlapping_agents():
    winners = set()
    for seed in range(20):
        env = placed_env()
        env.np_random = np.random.default_rng(seed)
        env.step({"agent_0": 2, "agent_1": 2})
        positions = [env.agent_states[a].position for a in env.agents]
        assert len(set(positions)) == 2
        assert positions.count((3, 3)) == 1
        winners.add(
            next(a for a in env.agents if env.agent_states[a].position == (3, 3))
        )
    assert winners == {"agent_0", "agent_1"}


def test_head_on_swap_is_blocked():
    env = placed_env()
    env.agent_states["agent_1"].position = (3, 3)
    env.step({"agent_0": 2, "agent_1": 2})
    assert env.agent_states["agent_0"].position == (2, 3)
    assert env.agent_states["agent_1"].position == (3, 3)


def test_ghost_mode_allows_contention_and_leaving_a_shared_cell():
    env = placed_env(ghost_mode=True)
    env.step({"agent_0": 2, "agent_1": 2})
    assert [s.position for s in env.agent_states.values()] == [(3, 3), (3, 3)]
    env.step({"agent_0": 2, "agent_1": 6})
    assert env.agent_states["agent_0"].position == (4, 3)
    assert env.agent_states["agent_1"].position == (3, 3)
    assert env.grid.get(3, 3) is None
    marker = env._observation_grid().get(3, 3)
    assert marker.color == env.agent_states["agent_1"].color


def test_ghost_mode_allows_head_on_swap():
    env = placed_env(ghost_mode=True)
    env.agent_states["agent_1"].position = (3, 3)
    env.step({"agent_0": 2, "agent_1": 2})
    assert env.agent_states["agent_0"].position == (3, 3)
    assert env.agent_states["agent_1"].position == (2, 3)


def test_non_ghost_passable_object_retains_source_overlap_rule():
    env = placed_env(ghost_mode=False)
    env.grid.set(3, 3, Floor())
    env.agent_states["agent_1"].position = (3, 3)
    env.step({"agent_0": 2, "agent_1": 6})
    assert [s.position for s in env.agent_states.values()] == [(3, 3), (3, 3)]
    assert env.grid.get(3, 3).type == "floor"


def test_ghost_mode_does_not_allow_dropping_onto_peer():
    env = placed_env(ghost_mode=True)
    env.agent_states["agent_1"].position = (3, 3)
    env.agent_states["agent_0"].carrying = Ball("blue")
    env.step({"agent_0": 4, "agent_1": 6})
    assert env.agent_states["agent_0"].carrying is not None
    assert env.grid.get(3, 3) is None


def test_wall_collision():
    env = placed_env()
    env.grid.set(3, 3, Wall())
    env.step({"agent_0": 2, "agent_1": 2})
    assert env.agent_states["agent_0"].position == (2, 3)
    assert env.agent_states["agent_1"].position == (4, 3)


def test_individual_success_does_not_end_other_agents():
    env = placed_env()
    env.agent_states["agent_0"].position = (6, 7)
    obs, rewards, term, trunc, infos = env.step({"agent_0": 2, "agent_1": 6})
    assert (
        set(obs)
        == set(rewards)
        == set(term)
        == set(trunc)
        == set(infos)
        == set(env.possible_agents)
    )
    assert term == {"agent_0": True, "agent_1": False}
    assert rewards["agent_0"] == 1 - 0.9 / env.max_steps
    assert rewards["agent_1"] == 0
    assert not any(trunc.values())
    assert env.agents == ["agent_1"]
    assert env.grid.get(7, 7).type == "goal"
    env.agent_states["agent_1"].position = (6, 7)
    env.agent_states["agent_1"].direction = 0
    obs, rewards, term, trunc, _ = env.step({"agent_1": 2})
    assert list(obs) == ["agent_1"] and term["agent_1"]
    assert env.agents == []
    assert env.step({}) == ({}, {}, {}, {}, {})
    with pytest.raises(ValueError):
        env.step({"agent_0": 2})


def test_simultaneous_goal_arrivals_are_individually_rewarded():
    env = placed_env()
    env.agent_states["agent_0"].position = (6, 7)
    env.agent_states["agent_1"].position = (7, 6)
    env.agent_states["agent_1"].direction = 1
    _, rewards, term, _, _ = env.step({a: 2 for a in env.agents})
    assert all(term.values()) and all(r > 0 for r in rewards.values())
    assert env.agents == []


def test_lava_terminates_only_the_entering_agent():
    env = placed_env()
    env.grid.set(3, 3, Lava())
    _, rewards, term, _, _ = env.step({"agent_0": 2, "agent_1": 6})
    assert term == {"agent_0": True, "agent_1": False}
    assert rewards == {"agent_0": 0, "agent_1": 0}


def test_timeout_keeps_final_positions_visible():
    env = placed_env(max_steps=1, render_mode="rgb_array")
    obs, _, term, trunc, _ = env.step({a: 6 for a in env.agents})
    assert all(trunc.values()) and not any(term.values())
    assert env.agents == []
    assert env._visible_agents == env.possible_agents
    assert data_equivalence(obs["agent_0"], env.observe("agent_0"))
    assert env.render().shape == (288, 288, 3)


def test_goal_on_final_step_is_termination_not_truncation():
    env = placed_env(max_steps=1)
    env.agent_states["agent_0"].position = (6, 7)
    _, rewards, term, trunc, _ = env.step({"agent_0": 2, "agent_1": 6})
    assert term == {"agent_0": True, "agent_1": False}
    assert trunc == {"agent_0": False, "agent_1": True}
    assert rewards["agent_0"] == pytest.approx(0.1)


@pytest.mark.parametrize("reward_decay", [True, False])
def test_reward_decay_can_be_disabled(reward_decay):
    env = placed_env(max_steps=10, reward_decay=reward_decay)
    env.agent_states["agent_0"].position = (6, 7)
    _, rewards, term, _, _ = env.step({"agent_0": 2, "agent_1": 6})
    assert term["agent_0"]
    assert rewards["agent_0"] == pytest.approx(0.91 if reward_decay else 1.0)


@pytest.mark.parametrize(
    "actions",
    [
        {},
        {"agent_0": 2},
        {"agent_0": 2, "agent_1": 2, "extra": 2},
        {"agent_0": 7, "agent_1": 2},
    ],
)
def test_bad_joint_action_is_atomic(actions):
    env = placed_env()
    states = copy.deepcopy(env.agent_states)
    random_state = copy.deepcopy(env.np_random.bit_generator.state)
    with pytest.raises(ValueError):
        env.step(actions)
    assert env.step_count == 0 and env.agent_states == states
    assert env.np_random.bit_generator.state == random_state


def test_pickup_drop_and_toggle_keep_per_agent_inventory():
    env = placed_env()
    env.grid.set(3, 3, Key("red"))
    env.step({"agent_0": 3, "agent_1": 6})
    first, second = (env.agent_states[a] for a in env.possible_agents)
    assert isinstance(first.carrying, Key) and second.carrying is None
    assert env.grid.get(3, 3) is None
    env.grid.set(3, 3, Door("red", is_locked=True))
    env.step({"agent_0": 5, "agent_1": 6})
    assert env.grid.get(3, 3).is_open
    env.grid.set(3, 3, None)
    env.step({"agent_0": 4, "agent_1": 6})
    assert first.carrying is None
    assert isinstance(env.grid.get(3, 3), Key)


def test_two_agents_cannot_pick_up_the_same_object():
    env = placed_env()
    env.grid.set(3, 3, Ball("blue"))
    env.step({a: 3 for a in env.agents})
    assert sum(s.carrying is not None for s in env.agent_states.values()) == 1
    assert env.grid.get(3, 3) is None


def test_agent_cannot_drop_onto_another_agent():
    env = placed_env()
    env.agent_states["agent_1"].position = (3, 3)
    env.agent_states["agent_0"].carrying = Ball("blue")
    env.step({"agent_0": 4, "agent_1": 6})
    assert env.agent_states["agent_0"].carrying is not None
    assert env.grid.get(3, 3) is None


@pytest.mark.parametrize("direction", range(4))
def test_peer_is_encoded_in_egocentric_observation(direction):
    env = placed_env()
    first, second = (env.agent_states[a] for a in env.possible_agents)
    first.direction = direction
    offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dx, dy = offsets[direction]
    second.position = (first.position[0] + dx, first.position[1] + dy)
    second.direction = 1
    observation = env.observe("agent_0")
    assert tuple(observation["image"][3, 5]) == (
        OBJECT_TO_IDX["agent"],
        COLOR_TO_IDX[second.color],
        (1 - direction + 3) % 4,
    )
    assert env.observation_space("agent_0").contains(observation)


def test_walls_occlude_peers_and_observe_does_not_mutate_world():
    env = placed_env()
    env.grid.vert_wall(3, 1, 7)
    before = env.grid.encode().copy()
    observation = env.observe("agent_0")
    assert not np.any(observation["image"][:, :, 0] == OBJECT_TO_IDX["agent"])
    np.testing.assert_array_equal(before, env.grid.encode())
    observation["image"][:] = 0
    assert np.any(env.observe("agent_0")["image"])


@pytest.mark.parametrize("env_class", [EmptyParallelEnv, ClutteredParallelEnv])
def test_starts_are_distinct_reachable_and_seeded(env_class):
    env = env_class(ghost_mode=False, ensure_reachable=True)
    for seed in range(25):
        obs, _ = env.reset(seed=seed)
        first_states = copy.deepcopy(env.agent_states)
        positions = [s.position for s in env.agent_states.values()]
        assert len(positions) == len(set(positions))
        assert all(p in env._reachable() for p in positions)
        assert all(p != (env.size - 2, env.size - 2) for p in positions)
        second_obs, _ = env.reset(seed=seed)
        assert data_equivalence(obs, second_obs)
        assert env.agent_states == first_states
        walls = sum(cell is not None and cell.type == "wall" for cell in env.grid.grid)
        assert walls == 4 * env.size - 4 + env.n_clutter


@pytest.mark.parametrize("env_class", [EmptyParallelEnv, ClutteredParallelEnv])
def test_default_source_rules_without_reachability_filter(env_class, monkeypatch):
    env = env_class()
    assert env.max_steps == 100
    assert env.ghost_mode and env.reward_decay
    assert not env.ensure_reachable

    def unexpected_filter():
        raise AssertionError("Source defaults must not filter unreachable starts")

    monkeypatch.setattr(env, "_reachable", unexpected_filter)
    env.reset(seed=0)
    assert all(s.direction == 0 for s in env.agent_states.values())
    env.step({a: 1 for a in env.agents})
    env.reset(seed=0)
    # Intentional API adaptation: reset does not inherit prior episode headings.
    assert all(s.direction == 0 for s in env.agent_states.values())


def test_default_spawn_may_overlap_on_goal_without_awarding_reward(monkeypatch):
    env = EmptyParallelEnv()

    def goal_position(accepts, max_tries=100000):
        assert accepts((7, 7))
        return (7, 7)

    monkeypatch.setattr(env, "_sample_position", goal_position)
    env.reset(seed=0)
    assert [s.position for s in env.agent_states.values()] == [(7, 7), (7, 7)]
    _, rewards, terminated, _, _ = env.step({a: 6 for a in env.agents})
    assert not any(rewards.values()) and not any(terminated.values())
    assert env.grid.get(7, 7).type == "goal"


def test_clutter_density_uses_interior_area():
    assert ClutteredParallelEnv().n_clutter == int(0.15 * 9 * 9)
    assert ClutteredParallelEnv(size=7, clutter_density=0.2).n_clutter == 5
    assert ClutteredParallelEnv(n_clutter=0).n_clutter == 0
    with pytest.raises(ValueError, match="not both"):
        ClutteredParallelEnv(n_clutter=0, clutter_density=0)


@pytest.mark.parametrize(
    "density", [-0.1, 1.1, float("nan"), float("inf"), True, "0.1"]
)
def test_invalid_density(density):
    with pytest.raises(ValueError, match="clutter_density"):
        ClutteredParallelEnv(clutter_density=density)


@pytest.mark.parametrize(
    "flag", ["ghost_mode", "reward_decay", "randomize_goal", "ensure_reachable"]
)
def test_flags_require_boolean(flag):
    with pytest.raises(ValueError, match=flag):
        EmptyParallelEnv(**{flag: "false"})


def test_random_goal_is_seeded_and_does_not_replace_clutter():
    env = ClutteredParallelEnv(size=7, n_clutter=5, randomize_goal=True)
    goals = set()
    for seed in range(10):
        observation, _ = env.reset(seed=seed)
        grid = env.grid.encode()
        goal = env.goal_pos
        goals.add(goal)
        assert env.grid.get(*goal).type == "goal"
        assert sum(c is not None and c.type == "wall" for c in env.grid.grid) == 29
        repeated, _ = env.reset(seed=seed)
        assert env.goal_pos == goal
        np.testing.assert_array_equal(grid, env.grid.encode())
        assert data_equivalence(observation, repeated)
    assert len(goals) > 1


def test_rejection_sampling_has_a_hard_attempt_limit():
    env = EmptyParallelEnv()
    attempts = []
    with pytest.raises(RuntimeError, match="3 attempts"):
        env._sample_position(lambda p: attempts.append(p), max_tries=3)
    assert len(attempts) == 3


def test_dense_grid_capacity_accounts_for_ghost_mode_and_goal_starts():
    assert ClutteredParallelEnv(size=5, n_clutter=8).n_clutter == 8
    with pytest.raises(ValueError, match="Not enough cells"):
        ClutteredParallelEnv(size=5, n_clutter=8, ghost_mode=False)
    with pytest.raises(ValueError, match="Not enough cells"):
        ClutteredParallelEnv(size=5, n_clutter=8, ensure_reachable=True)


_REFERENCE_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "marlgrid_navigation.json").read_text(
        encoding="utf-8"
    )
)["cases"]


@pytest.mark.parametrize("case", _REFERENCE_CASES, ids=lambda case: case["name"])
def test_pinned_marlgrid_transition_reference(case):
    """Compare with outputs of original sources, not another local implementation.

    RNG is standardized by the capture harness; image/visibility and legacy Gym
    seed sequences are deliberately outside this comparison. See fixture metadata.
    """
    env_class = EmptyParallelEnv if case["family"] == "empty" else ClutteredParallelEnv
    env = env_class(**case["kwargs"])
    try:
        env.reset(seed=case["seed"])
        walls = [
            [x, y]
            for x in range(env.size)
            for y in range(env.size)
            if isinstance(env.grid.get(x, y), Wall)
        ]
        assert walls == case["walls"]
        assert list(env.goal_pos) == case["goal"]
        if case["placement"] is not None:
            for agent, (x, y, direction) in zip(env.possible_agents, case["placement"]):
                env.agent_states[agent].position = (x, y)
                env.agent_states[agent].direction = direction

        def states(completed):
            return [
                [
                    *env.agent_states[a].position,
                    env.agent_states[a].direction,
                    a in completed,
                ]
                for a in env.possible_agents
            ]

        completed = set()
        assert states(completed) == case["initial"]
        for row in case["steps"]:
            actions = {
                a: row["actions"][i]
                for i, a in enumerate(env.possible_agents)
                if a in env.agents
            }
            obs, rewards, term, trunc, _ = env.step(actions)
            completed.update(a for a, finished in term.items() if finished)
            assert states(completed) == row["states"]
            assert [rewards.get(a, 0) for a in env.possible_agents] == pytest.approx(
                row["rewards"]
            )
            assert (not env.agents) == row["done"]
            for a in actions:
                assert trunc[a] == (env.step_count >= env.max_steps and not term[a])
                assert env.observation_space(a).contains(obs[a])
    finally:
        env.close()


def test_pickled_mid_episode_continues_identically():
    env = placed_env()
    env.step({a: 1 for a in env.agents})
    restored = pickle.loads(pickle.dumps(env))
    for _ in range(20):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        assert data_equivalence(env.step(actions), restored.step(actions))
        assert env.agent_states == restored.agent_states


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 4},
        {"size": 5.5},
        {"num_agents": 0},
        {"num_agents": 7},
        {"num_agents": True},
        {"agent_view_size": 4},
        {"max_steps": 0},
        {"max_steps": 2.5},
        {"render_mode": "human"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EmptyParallelEnv(**kwargs)


@pytest.mark.parametrize("n_clutter", [-1, 9, 0.5])
def test_invalid_clutter(n_clutter):
    with pytest.raises(ValueError):
        ClutteredParallelEnv(size=5, num_agents=2, n_clutter=n_clutter)


def test_single_agent_import_does_not_require_pettingzoo():
    code = """
import importlib.abc
import sys
class BlockPettingZoo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pettingzoo" or fullname.startswith("pettingzoo."):
            raise ModuleNotFoundError("Blocked optional dependency", name="pettingzoo")
sys.meta_path.insert(0, BlockPettingZoo())
import gymnasium as gym
import minigrid
from minigrid.multiagent import EmptyParallelEnv
env = gym.make("MiniGrid-Empty-5x5-v0")
env.reset(seed=0)
env.step(2)
env.close()
try:
    EmptyParallelEnv()
except ImportError as error:
    assert "minigrid[multiagent]" in str(error)
else:
    raise AssertionError("Expected an actionable optional-dependency error")
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
