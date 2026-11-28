from __future__ import annotations

import pickle
from collections import deque

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env, data_equivalence

from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.envs.windy import WindyEnv
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper


def windy(probability=1.0, **kwargs):
    env = WindyEnv(wind_probabilities=[probability] * 8, **kwargs)
    env.reset(seed=42)
    env.agent_pos = (3, 4)
    return env


@pytest.mark.parametrize("action", range(7))
def test_wind_follows_every_action_once(action):
    env = windy()
    obs, reward, terminated, truncated, _ = env.step(action)
    assert env.agent_pos == ((4, 3) if action == 2 else (3, 3))
    assert env.agent_dir == ({0: 3, 1: 1}.get(action, 0))
    assert env.step_count == 1
    assert reward == 0 and not terminated and not truncated
    assert data_equivalence(obs, env.gen_obs())


@pytest.mark.parametrize("action", range(7))
def test_zero_wind_and_orientation(action):
    env = windy(0)
    rng_before = pickle.dumps(env.np_random.bit_generator.state)
    env.step(action)
    assert env.agent_pos == ((4, 4) if action == 2 else (3, 4))
    assert pickle.dumps(env.np_random.bit_generator.state) == rng_before


def test_source_column_determines_wind():
    probabilities = [0.0] * 8
    probabilities[3] = 1.0
    env = WindyEnv(wind_probabilities=probabilities)
    env.reset(seed=0)
    env.agent_pos = (3, 4)
    env.step(2)
    assert env.agent_pos == (4, 3)
    env.step(2)
    assert env.agent_pos == (5, 3)


def test_configuration_is_not_aliased():
    probabilities = np.zeros(8)
    env = WindyEnv(wind_probabilities=probabilities)
    probabilities[:] = 1
    assert env.wind_probabilities == (0.0,) * 8


@pytest.mark.parametrize("barrier", ["boundary", "internal"])
def test_wind_respects_walls(barrier):
    env = windy()
    if barrier == "boundary":
        env.agent_pos = (3, 1)
    else:
        env.grid.set(3, 3, Wall())
    previous = env.agent_pos
    env.step(6)
    assert env.agent_pos == previous


def test_blocked_action_can_still_be_followed_by_wind():
    env = windy()
    env.grid.set(4, 4, Wall())
    env.step(2)
    assert env.agent_pos == (3, 3)


@pytest.mark.parametrize("via_wind", [False, True])
@pytest.mark.parametrize("terminal_object", [Goal, Lava])
def test_terminal_cell(via_wind, terminal_object):
    env = windy()
    target = (3, 3) if via_wind else (4, 4)
    env.grid.set(*target, terminal_object())
    obs, reward, terminated, truncated, _ = env.step(6 if via_wind else 2)
    assert env.agent_pos == target
    assert terminated and not truncated
    assert reward == (env._reward() if terminal_object is Goal else 0)
    assert data_equivalence(obs, env.gen_obs())


def test_last_step_includes_wind_and_preserves_both_end_flags():
    env = windy(max_steps=1)
    env.grid.set(3, 3, Goal())
    _, reward, terminated, truncated, _ = env.step(6)
    assert terminated and truncated
    assert reward == pytest.approx(0.1)
    assert env.agent_pos == (3, 3) and env.step_count == 1


def test_timeout_without_success():
    env = windy(max_steps=1)
    _, reward, terminated, truncated, _ = env.step(6)
    assert not terminated and truncated and reward == 0
    assert env.agent_pos == (3, 3)


def test_human_render_sees_only_final_state(monkeypatch):
    env = windy(render_mode="human")
    frames = []
    monkeypatch.setattr(env, "render", lambda: frames.append(env.agent_pos))
    env.step(2)
    assert frames == [(4, 3)]
    env.close()


@pytest.mark.parametrize("invalid_action", [-1, 7, 1.5, "forward", None])
def test_invalid_action_does_not_advance_state(invalid_action):
    env = windy()
    before = (env.agent_pos, env.agent_dir, env.step_count)
    with pytest.raises(ValueError):
        env.step(invalid_action)
    assert (env.agent_pos, env.agent_dir, env.step_count) == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 4},
        {"height": 4},
        {"width": 8.0},
        {"height": True},
        {"two_rooms": True, "width": 6},
        {"max_steps": 0},
        {"max_steps": -1},
        {"max_steps": 2.5},
        {"max_steps": True},
        {"wind_probabilities": []},
        {"wind_probabilities": [0] * 7},
        *({"wind_probabilities": [p] * 8} for p in [-0.1, 1.1, np.nan, np.inf, "0.5"]),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        WindyEnv(**kwargs)


@pytest.mark.parametrize(
    "env_id", ["MiniGrid-Windy-8x8-v0", "MiniGrid-Windy-TwoRooms-12x9-v0"]
)
def test_gymnasium_registration_and_checker(env_id):
    env = gym.make(env_id).unwrapped
    try:
        check_env(env)
        assert env.observation_space.contains(env.reset(seed=12)[0])
    finally:
        env.close()


@pytest.mark.parametrize("two_rooms", [False, True])
def test_layout_has_reachable_goal(two_rooms):
    env = WindyEnv(width=12, height=9, two_rooms=two_rooms)
    env.reset(seed=0)
    seen = {env.agent_pos}
    frontier = deque(seen)
    while frontier:
        x, y = frontier.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            pos = (x + dx, y + dy)
            cell = env.grid.get(*pos)
            if pos not in seen and (cell is None or cell.can_overlap()):
                seen.add(pos)
                frontier.append(pos)
    assert (10, 4) in seen
    if two_rooms:
        assert env.grid.get(6, 4) is None
        assert all(env.grid.get(6, y).type == "wall" for y in range(1, 8) if y != 4)


def test_seeded_rollouts_and_pickling():
    first = windy(0.45)
    second = pickle.loads(pickle.dumps(first))
    for action in np.random.default_rng(11).integers(0, 7, size=200):
        result = first.step(action)
        assert data_equivalence(result, second.step(action))
        assert first.agent_pos == second.agent_pos
        if result[2] or result[3]:
            assert data_equivalence(first.reset(), second.reset())
    first.reset(seed=13)
    second.reset(seed=13)
    assert data_equivalence(first.step(2), second.step(2))


def test_step_requires_reset():
    with pytest.raises(ResetNeeded):
        WindyEnv().step(2)


@pytest.mark.parametrize("two_rooms", [False, True])
def test_zero_wind_matches_base_transitions(two_rooms):
    env = WindyEnv(two_rooms=two_rooms, wind_probabilities=[0] * 8, max_steps=75)
    env.reset(seed=21)
    reference = pickle.loads(pickle.dumps(env))
    for action in np.random.default_rng(21).integers(0, 7, size=300):
        result = env.step(action)
        expected = MiniGridEnv.step(reference, action)
        assert data_equivalence(result[0], expected[0])
        assert result[1:] == expected[1:]
        assert env.agent_pos == tuple(reference.agent_pos)
        if result[2] or result[3]:
            assert data_equivalence(env.reset(), reference.reset())


def test_sync_vector_environment():
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("MiniGrid-Windy-8x8-v0") for _ in range(3)]
    )
    try:
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        for _ in range(20):
            obs, rewards, terminations, truncations, _ = env.step([0, 1, 2])
            assert env.observation_space.contains(obs)
            assert rewards.shape == terminations.shape == truncations.shape == (3,)
    finally:
        env.close()


def test_empirical_probability_and_different_seeds():
    outcomes = []
    for seed in (17, 29):
        env = WindyEnv(wind_probabilities=[0.35] * 8, max_steps=10001)
        env.reset(seed=seed)
        pushes = []
        for _ in range(10000):
            env.agent_pos = (3, 4)
            env.step(6)
            pushes.append(env.agent_pos == (3, 3))
        assert abs(np.mean(pushes) - 0.35) < 0.02
        outcomes.append(pushes)
    assert outcomes[0] != outcomes[1]


@pytest.mark.parametrize("wrapper", [FullyObsWrapper, RGBImgObsWrapper])
def test_wrapped_observation_matches_post_wind_state(wrapper):
    env = wrapper(windy())
    obs, *_ = env.step(2)
    assert env.unwrapped.agent_pos == (4, 3)
    assert env.observation_space.contains(obs)
    assert data_equivalence(obs, env.observation(env.unwrapped.gen_obs()))
    env.close()
