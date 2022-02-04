#!/usr/bin/env python3

import time
import random
import gym
import gym_minigrid

# Load the gym environment
env = gym.make('MiniGrid-Empty-8x8-v0')
env.reset()

for i in range(0, 100):
    print("step {}".format(i))

    # Pick a random action
    action = random.randint(0, env.action_space.n - 1)

    obs, reward, done, info = env.step(action)

    env.render()

    time.sleep(0.05)

# Test the close method
env.close()
