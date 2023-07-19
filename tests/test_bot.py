from __future__ import annotations

import gymnasium as gym
import numpy as np

from minigrid.utils.bot import BabyAIBot


def test_bot():
    # get all babyai envs
    babyai_envs = []
    for k_i in gym.envs.registry.keys():
        if k_i.split("-")[0] == "BabyAI":
            babyai_envs.append(k_i)

    # random select one env
    env = gym.make(np.random.choice(babyai_envs), render_mode="human")

    # reset env
    env.reset()
    env.render()

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
