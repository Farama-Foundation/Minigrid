import math
import gym

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def _step(self, action):

        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (env.agentPos, env.agentDir, action)

        # Get the count for this (s,a) pair
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this (s,a) pair
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus

        return obs, reward, done, info

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def _step(self, action):

        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (env.agentPos)

        # Get the count for this key
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this key
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus

        return obs, reward, done, info
