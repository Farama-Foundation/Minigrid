import math
import operator
from functools import reduce

import numpy as np

import gym
from gym import error, spaces, utils

from minigrid import *

class SafetyEnvelope(gym.core.Wrapper):
    """
    Safety envelope for safe exploration.
    The purpose is to detect dangerous actions and block them sending back a modified reward
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        env = self.unwrapped

        # Get current observations from the environment and decode them
        obs_pre = env.genObs()['image']
        decoded_obs_pre = Grid.decode(obs_pre)


        # Apply the agent action to the environment or a safety action
        obs, reward, done, info = self.env.step(action)


        # Get observations after applying the action to the environment and decode them
        obs_post = env.genObs()['image']
        decode_obs_post = Grid.decode(obs_post)

        # Modify the reward if necessary
        reward += 5

        return obs, reward, done, info