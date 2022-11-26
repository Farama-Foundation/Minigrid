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
for i in range(5):
    yPos = np.random.randint(2, 18)
    xPos = np.random.randint(10, 18)
    direction = 2 if np.random.random() > 0.5 else 0
    agents.append(PedAgent(i, (xPos, yPos), direction))
env.addAgents(agents)

env.reset()

for i in range(0, 20):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    time.sleep(0.05)


# Test the close method
env.close()
