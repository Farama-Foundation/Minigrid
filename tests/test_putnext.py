from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from minigrid.core.constants import OBJECT_TO_IDX


@pytest.mark.parametrize("size,objects", [(5, 2), (6, 3), (7, 4)])
@pytest.mark.parametrize("carrying", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_reset_observation(size, objects, carrying, seed):
    suffix = "Carrying" if carrying else ""
    env = gym.make(f"BabyAI-PutNextS{size}N{objects}{suffix}-v0")
    try:
        obs, info = env.reset(seed=seed)
        view_size = env.unwrapped.agent_view_size
        expected = (
            env.unwrapped.carrying.encode()
            if carrying
            else (OBJECT_TO_IDX["empty"], 0, 0)
        )
        # The agent's own cell represents the object in its inventory.
        np.testing.assert_array_equal(
            obs["image"][view_size // 2, view_size - 1], expected
        )
        assert info == {}
        assert env.observation_space.contains(obs)

        # A no-op does not change the grid, pose, or inventory, so its
        # observation must agree with the one returned by reset.
        next_obs, reward, terminated, truncated, _ = env.step(
            env.unwrapped.actions.done
        )
        np.testing.assert_array_equal(obs["image"], next_obs["image"])
        assert obs["direction"] == next_obs["direction"]
        assert obs["mission"] == next_obs["mission"]
        assert reward == 0
        assert not terminated and not truncated
    finally:
        env.close()


@pytest.mark.parametrize("size,objects,seed", [(5, 2, 2), (6, 3, 20), (7, 4, 33)])
def test_first_action_drop_completes_mission(size, objects, seed):
    env = gym.make(f"BabyAI-PutNextS{size}N{objects}Carrying-v0")
    try:
        env.reset(seed=seed)
        u = env.unwrapped
        target = u.instrs.desc_fixed.obj_set[0]
        # These layouts put an empty drop cell next to the requested target.
        assert u.grid.get(*u.front_pos) is None
        assert np.abs(u.front_pos - target.cur_pos).sum() == 1
        held = u.carrying

        obs, reward, terminated, truncated, _ = env.step(u.actions.drop)

        assert u.carrying is None
        assert u.grid.get(*u.front_pos) is held
        assert terminated and not truncated
        assert reward == 1 - 0.9 / u.max_steps
        np.testing.assert_array_equal(
            obs["image"][u.agent_view_size // 2, u.agent_view_size - 1],
            (OBJECT_TO_IDX["empty"], 0, 0),
        )
    finally:
        env.close()
