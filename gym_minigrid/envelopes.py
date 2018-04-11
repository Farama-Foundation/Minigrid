import math
import operator
import time

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

        # Stores history of the last N observation / proposed_actions
        self.proposed_history = collections.deque(N*[(None, None)], N)

        # Stores history of the last N observation / applied_actions
        self.actual_history = collections.deque(N * [(None, None)], N)

    def step(self, action):

        # Get current observations from the environment and decode them
        current_obs = self.env.genObs()['image']
        current_obs = Grid.decode(current_obs)

        proposed_action = action

        # If we go forward, blocking is relevant.
        # Action:
        #         Turn left, turn right, move forward
        #         left = 0
        #         right = 1
        #         forward = 2
        #         # Toggle/pick up/activate object
        #         toggle = 3
        #         # Wait/stay put/do nothing
        #         wait = 4
        # Store the observation-action tuple in the history
        self.proposed_history.append((current_obs, proposed_action))

        safe_action = proposed_action

        if proposed_action == 2:
            if self.blocker(current_obs, proposed_action):
                #self.env.displayAlert()
                obs, reward, done, info = self.env.step(4)
                safe_action = 4
                obs, reward, done, info = self.env.step(safe_action)
                reward = -1
            else:
                obs, reward, done, info = self.env.step(safe_action)
        else:
            obs, reward, done, info = self.env.step(safe_action)

        self.actual_history.append((current_obs, safe_action))

        # Apply the agent action to the environment or a safety action

        mod_reward = reward

        # Create a window to render into
        self.env.render('human')

        return obs, mod_reward, done, info

    def blocker(self, observation, action):
        # Check if tile in direction of action is type catastrophe, then say: NO!
        # Display alert if this is the case.
        if (isinstance(observation.get(1,1), Water)):
            return True
        return False


