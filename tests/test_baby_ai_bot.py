from __future__ import annotations

import gymnasium as gym
import pytest

from minigrid.utils.baby_ai_bot import BabyAIBot

# see discussion starting here: https://github.com/Farama-Foundation/Minigrid/pull/381#issuecomment-1646800992
broken_bonus_envs = {
    "BabyAI-PutNextS5N2Carrying-v0",
    "BabyAI-PutNextS6N3Carrying-v0",
    "BabyAI-PutNextS7N4Carrying-v0",
    "BabyAI-KeyInBox-v0",
}

# get all babyai envs (except the broken ones)
babyai_envs = []
for k_i in gym.envs.registry.keys():
    if k_i.split("-")[0] == "BabyAI":
        if k_i not in broken_bonus_envs:
            babyai_envs.append(k_i)


@pytest.mark.parametrize("env_id", babyai_envs)
def test_bot(env_id):
    """The BabyAI Bot should be able to solve all BabyAI environments,
    allowing us therefore to generate demonstrations.
    """
    env = gym.make(env_id)
    num_steps = getattr(env, "max_steps", 240)

    try:
        for seed in (0, 1):
            env.reset(seed=seed)
            expert = BabyAIBot(env)

            last_action = None
            terminated = False
            for _step in range(num_steps):
                action = expert.replan(last_action)
                obs, reward, terminated, truncated, info = env.step(action)
                last_action = action

                if terminated:
                    break

            assert (
                terminated
            ), f"BabyAIBot failed to solve {env_id} on seed {seed} within {num_steps} steps"
    finally:
        env.close()
