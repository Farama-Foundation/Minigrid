from __future__ import annotations

import gymnasium as gym
import pytest

import minigrid  # noqa: F401
from tests.utils import assert_equals


@pytest.mark.parametrize(
    "env_id",
    ["BabyAI-MixedTrainLocal-v0", "BabyAI-MixedTestLocal-v0"],
)
@pytest.mark.parametrize("seed", [0, 1, 4, 11, 21])
def test_mixed_tasks_reproduce_seeded_episode(env_id, seed):
    env_a = gym.make(env_id)
    env_b = gym.make(env_id)

    try:
        initial = env_a.reset(seed=seed)

        # Independent environments must reproduce the same episode.
        assert_equals(initial, env_b.reset(seed=seed))

        # An intervening episode must not affect a later seeded reset.
        env_a.reset(seed=seed + 100)
        assert_equals(initial, env_a.reset(seed=seed))
    finally:
        env_a.close()
        env_b.close()
