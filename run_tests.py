#!/usr/bin/env python3

import random
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX

# Test specifically importing a specific environment
from gym_minigrid.envs import DoorKeyEnv

# Test importing wrappers
from gym_minigrid.wrappers import *

##############################################################################

print('%d environments registered' % len(env_list))

for env_idx, env_name in enumerate(env_list):
    print('testing {} ({}/{})'.format(env_name, env_idx+1, len(env_list)))

    # Load the gym environment
    env = gym.make(env_name)
    env.max_steps = min(env.max_steps, 200)
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

        # Validate the agent position
        assert env.agent_pos[0] < env.width
        assert env.agent_pos[1] < env.height

        # Test observation encode/decode roundtrip
        img = obs['image']
        grid, vis_mask = Grid.decode(img)
        img2 = grid.encode(vis_mask=vis_mask)
        assert np.array_equal(img, img2)

        # Test the env to string function
        str(env)

        # Check that the reward is within the specified range
        assert reward >= env.reward_range[0], reward
        assert reward <= env.reward_range[1], reward

        if done:
            num_episodes += 1
            env.reset()

        env.render('rgb_array')

    # Test the close method
    env.close()

    env = gym.make(env_name)
    env = ReseedWrapper(env)
    for _ in range(10):
        env.reset()
        env.step(0)
        env.close()

    env = gym.make(env_name)
    env = ImgObsWrapper(env)
    env.reset()
    env.step(0)
    env.close()

    # Test the fully observable wrapper
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env.reset()
    obs, _, _, _ = env.step(0)
    assert obs['image'].shape == env.observation_space.spaces['image'].shape
    env.close()

    # RGB image observation wrapper
    env = gym.make(env_name)
    env = RGBImgPartialObsWrapper(env)
    env.reset()
    obs, _, _, _ = env.step(0)
    assert obs['image'].mean() > 0
    env.close()

    env = gym.make(env_name)
    env = FlatObsWrapper(env)
    env.reset()
    env.step(0)
    env.close()

    env = gym.make(env_name)
    env = ViewSizeWrapper(env, 5)
    env.reset()
    env.step(0)
    env.close()

    # Test the wrappers return proper observation spaces.
    wrappers = [
        RGBImgObsWrapper,
        RGBImgPartialObsWrapper,
        OneHotPartialObsWrapper
    ]
    for wrapper in wrappers:
        env = wrapper(gym.make(env_name))
        obs_space, wrapper_name = env.observation_space, wrapper.__name__
        assert isinstance(
            obs_space, spaces.Dict
        ), "Observation space for {0} is not a Dict: {1}.".format(
            wrapper_name, obs_space
        )
        # This should not fail either
        ImgObsWrapper(env)
        env.reset()
        env.step(0)
        env.close()

##############################################################################

print('testing extra observations')
class EmptyEnvWithExtraObs(gym_minigrid.envs.EmptyEnv5x5):
    """
    Custom environment with an extra observation
    """
    def __init__(self) -> None:
        super().__init__()
        self.observation_space['size'] = spaces.Box(
            low=0,
            high=np.iinfo(np.uint).max,
            shape=(2,),
            dtype=np.uint
        )

    def reset(self):
        obs = super().reset()
        obs['size'] = np.array([self.width, self.height])
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs['size'] = np.array([self.width, self.height])
        return obs, reward, done, info

wrappers = [
    OneHotPartialObsWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    FullyObsWrapper,
]
for wrapper in wrappers:
    env1 = wrapper(EmptyEnvWithExtraObs())
    env2 = wrapper(gym.make('MiniGrid-Empty-5x5-v0'))

    env1.seed(0)
    env2.seed(0)

    obs1 = env1.reset()
    obs2 = env2.reset()
    assert 'size' in obs1
    assert obs1['size'].shape == (2,)
    assert (obs1['size'] == [5,5]).all()
    for key in obs2:
        assert np.array_equal(obs1[key], obs2[key])

    obs1, reward1, done1, _ = env1.step(0)
    obs2, reward2, done2, _ = env2.step(0)
    assert 'size' in obs1
    assert obs1['size'].shape == (2,)
    assert (obs1['size'] == [5,5]).all()
    for key in obs2:
        assert np.array_equal(obs1[key], obs2[key])

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

    grid, _ = Grid.decode(obs['image'])
    goal_visible = ('green', 'goal') in grid

    agent_sees_goal = env.agent_sees(*goal_pos)
    assert agent_sees_goal == goal_visible
    if done:
        env.reset()
