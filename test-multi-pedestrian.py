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
for i in range(50):
    xPos = np.random.randint(2, 18)
    yPos = np.random.randint(10, 90)
    direction = 1 if np.random.random() > 0.5 else 3
    prob = np.random.random()
    if prob < 0.1:
        speed = 1
    elif prob < 0.9:
        speed = 2
    else:
        speed = 3
    agents.append(PedAgent((xPos, yPos), direction, speed))
env.addAgents(agents)

env.reset()

for i in range(0, 100):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    time.sleep(0.05)


# Test the close method
env.close()
