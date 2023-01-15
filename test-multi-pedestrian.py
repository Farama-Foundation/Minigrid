import logging
import random
import time

import pytest

import gym
import numpy as np
import pickle

import gym_minigrid
from gym_minigrid.agents import PedAgent
from gym_minigrid.lib.MetricCollector import MetricCollector
from gym_minigrid.wrappers import *

logging.basicConfig(level=logging.INFO)

# from gym_minigrid.envs import MultiPedestrianEnv
# %matplotlib auto
# %load_ext autoreload
# %autoreload 2

# Load the gym environment
env = gym.make('MultiPedestrian-Empty-20x80-v0')
metricCollector = MetricCollector(env)
agents = []

possibleX = list(range(0, env.width))
possibleY = list(range(1, env.height - 1))
possibleCoordinates = []
for i in possibleX:
    for j in possibleY:
        possibleCoordinates.append((i, j))

logging.info(f"Number of possible coordinates is {len(possibleCoordinates)}")

for i in range(int(env.density * env.width * (env.height - 2))): # -2 from height to account for top and bottom
    randomIndex = np.random.randint(0, len(possibleCoordinates) - 1)
    pos = possibleCoordinates[randomIndex]
    direction = 2 if np.random.random() > 0.5 else 0
    agents.append(PedAgent(i, pos, direction))
    del possibleCoordinates[randomIndex]
env.addAgents(agents)

env.reset()

for i in range(1100):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    # env.render()

    if i % 100 == 0:
        logging.info(f"Completed step {i+1}")

    # time.sleep(0.05)

logging.info(env.getAverageSpeed())

stepStats = metricCollector.getStatistics()[0]
avgSpeed = sum(stepStats["xSpeed"]) / len(stepStats["xSpeed"])
logging.info("Average speed: " + str(avgSpeed))
volumeStats = metricCollector.getStatistics()[1]
avgVolume = sum(volumeStats) / len(volumeStats)
logging.info("Average volume: " + str(avgVolume))

dump = (avgSpeed, avgVolume)
with open(f"{env.DML}.{env.p_exchg}.{env.density}.pickle", "wb") as f:
    pickle.dump(dump, f)

# Test the close method
env.close()