import time
import logging
import gym
import pedgrid
from pedgrid.agents import *

env = gym.make('PedestrianEnv20x80-v0')
env.reset()

