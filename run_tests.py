#!/usr/bin/env python3

import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid

# Test specifically importing a specific environment
from gym_minigrid.envs import DoorKeyEnv

# Test importing wrappers
from gym_minigrid.wrappers import *

##############################################################################

print('%d environments registered' % len(env_list))

for envName in env_list:
    print('testing "%s"' % envName)

    # Load the gym environment
    env = gym.make(envName)
    env.reset()
    env.render('rgb_array')

    # Verify that the same seed always produces the same environment
    for i in range(0, 5):
        seed = 1337 + i
        env.seed(seed)
        grid1 = env.grid
        env.seed(seed)
        grid2 = env.grid
        assert grid1 == grid2

    env.reset()

    # Run for a few episodes
    num_episodes = 0
    while num_episodes < 5:
        # Pick a random action
        action = random.randint(0, env.action_space.n - 1)

        obs, reward, done, info = env.step(action)

        # Test observation encode/decode roundtrip
        img = obs['image']
        grid = Grid.decode(img)
        img2 = grid.encode()
        assert np.array_equal(img, img2)

        # Check that the reward is within the specified range
        assert reward >= env.reward_range[0], reward
        assert reward <= env.reward_range[1], reward

        if done:
            num_episodes += 1
            env.reset()

        env.render('rgb_array')

    env.close()

##############################################################################

print('testing agent_sees method')
env = gym.make('MiniGrid-DoorKey-6x6-v0')
goal_pos = (env.grid.width - 2, env.grid.height - 2)

# Test the "in" operator on grid objects
assert ('green', 'goal') in env.grid
assert ('blue', 'key') not in env.grid

# Test the env.agent_sees() function
env.reset()
for i in range(0, 500):
    action = random.randint(0, env.action_space.n - 1)
    obs, reward, done, info = env.step(action)
    goal_visible = ('green', 'goal') in Grid.decode(obs['image'])
    agent_sees_goal = env.agent_sees(*goal_pos)
    assert agent_sees_goal == goal_visible
    if done:
        env.reset()
