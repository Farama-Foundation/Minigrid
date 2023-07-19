from __future__ import annotations

import gymnasium as gym
import pytest

from minigrid.utils.baby_ai_bot import BabyAIBot

# get all babyai envs
babyai_envs = []
for k_i in gym.envs.registry.keys():
    if k_i.split("-")[0] == "BabyAI":
        babyai_envs.append(k_i)


@pytest.mark.parametrize("env_id", babyai_envs)
def test_bot(env_id):
    """
    The BabyAI Bot should be able to solve all BabyAI environments,
    allowing us therefore to generate demonstrations.
    """
    # Use the parameter env_id to make the environment
    env = gym.make(env_id)
    # env = gym.make(env_id, render_mode="human") # for visual debugging

    # reset env
    env.reset()

    # create expert bot
    expert = BabyAIBot(env)

    last_action = None
    while True:
        action = expert.replan(last_action)
        obs, reward, terminated, truncated, info = env.step(action)
        last_action = action
        env.render()

        if terminated:
            break

    env.close()
