import math
import operator
import collections
from functools import reduce

import numpy as np

import gym
from gym import error, spaces, utils

from minigrid import *

# Size of the history collection
N = 5

class SafetyEnvelope(gym.core.RewardWrapper):
    """
    Safety envelope for safe exploration.
    The purpose is to detect dangerous actions and block them sending back a modified reward
    """

    def __init__(self, env):
        super().__init__(env)

        # stores history of the last N observation / actions
        self.history = collections.deque(N*[(None, None)], N)

    def step(self, action):

        # Get current observations from the environment and decode them
        current_obs = self.env.genObs()['image']
        current_obs = Grid.decode(current_obs)

        suggested_action = action

        # Store the observation-action tuple in the history
        self.history.append((current_obs, suggested_action))


        # Create a window to render into
        self.env.render('human')

        # Apply the agent action to the environment or a safety action
        obs, reward, done, info = self.env.step(action)

        # Get observations after applying the action to the environment and decode them
        # obs_post = env.genObs()['image']
        # decode_obs_post = Grid.decode(obs_post)

        # Modify the reward if necessary
        # reward += 5

        return obs, reward, done, info


