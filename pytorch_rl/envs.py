import os
import numpy
import gym
from gym import spaces

try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
except:
    pass

def make_env(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)

        env.seed(seed + rank)

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        if isinstance(env.observation_space, spaces.Dict):
            env = FlatObsWrapper(env)

        return env

    return _thunk
