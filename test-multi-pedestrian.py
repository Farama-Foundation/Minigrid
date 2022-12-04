import time
import random
import gym
import numpy as np
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.agents import PedAgent
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
print(len(possibleCoordinates))
for i in range(int(density * width * height)):
    randomIndex = np.random.randint(0, len(possibleCoordinates) - 1)
    pos = possibleCoordinates[randomIndex]
    direction = 2 if np.random.random() > 0.5 else 0
    agents.append(PedAgent(i, pos, direction))
    del possibleCoordinates[randomIndex]
env.addAgents(agents)

def helloWorld(env):
    print("helloWorld")

env.subscribe("stepParallel1", helloWorld)

env.reset()

for i in range(0, 100):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    time.sleep(0.05)

print(env.getAverageSpeed())

# Test the close method
env.close()
