import time
import random
import gym
import numpy as np
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.agents import PedAgent
import logging


logging.basicConfig(level=logging.INFO)

# from gym_minigrid.envs import MultiPedestrianEnv
# %matplotlib auto
# %load_ext autoreload
# %autoreload 2

# Load the gym environment
env = gym.make('MultiPedestrian-Empty-20x80-v0')
agents = []
width = 30
height = 10
density = 0.5

possibleX = list(range(0, width))
possibleY = list(range(5, height + 5))
possibleCoordinates = []
for i in possibleX:
    for j in possibleY:
        possibleCoordinates.append((i, j))

logging.info(f"Number of possible coordinates is {len(possibleCoordinates)}")

for i in range(int(density * width * height)):
    randomIndex = np.random.randint(0, len(possibleCoordinates) - 1)
    pos = possibleCoordinates[randomIndex]
    direction = 2 if np.random.random() > 0.5 else 0
    agents.append(PedAgent(i, pos, direction))
    del possibleCoordinates[randomIndex]
env.addAgents(agents)

env.reset()

for i in range(1000):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    # env.render()

    if i % 100 == 0:
        logging.info(f"Completed step {i+1}")

    # time.sleep(0.05)

logging.info(env.getAverageSpeed())

# Test the close method
env.close()
