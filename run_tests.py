#!/usr/bin/env python3

import random
import gym
from gym_minigrid.register import envSet

print('%d environments registered' % len(envSet))

for envName in sorted(envSet):
    print('testing "%s"' % envName)

    # Load the gym environment
    env = gym.make(envName)
    env.reset()
    env.render('rgb_array')

    env.seed()
    env.reset()

    numActions = env.action_space.n
    for i in range(500):
        action = random.randint(0, numActions - 1)
        env.step(action)
        env.render('rgb_array')

    env.close()
