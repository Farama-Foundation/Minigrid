import math
import gym

class ExplBonus(gym.core.Wrapper):

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
