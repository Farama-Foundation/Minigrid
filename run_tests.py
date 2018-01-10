#!/usr/bin/env python3

import random
import gym
import numpy as np
from gym_minigrid.register import envSet
from gym_minigrid.minigrid import Grid

print('%d environments registered' % len(envSet))

for envName in sorted(envSet):
    print('testing "%s"' % envName)

    # Load the gym environment
    env = gym.make(envName)
    env.reset()
    env.render('rgb_array')

    env.seed()
    env.reset()

    # Run for a few episodes
    for i in range(5 * env.maxSteps):
        # Pick a random action
        action = random.randint(0, env.action_space.n - 1)

        obs, reward, done, info = env.step(action)

        # Test observation encode/decode roundtrip
        if type(obs) is np.ndarray:
            grid = Grid.decode(obs)
            obs2 = grid.encode()
            assert np.array_equal(obs2, obs)

        assert reward >= env.reward_range[0], reward
        assert reward <= env.reward_range[1], reward

        env.render('rgb_array')

    env.close()
