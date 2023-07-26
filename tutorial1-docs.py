import time
import logging
import gym
import gym_minigrid
from gym_minigrid.agents import *

env = gym.make('PedestrianEnv20x80-v0')
env.reset()

