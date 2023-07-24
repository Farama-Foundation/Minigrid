import time
import random
import gym
import numpy as np
import gym_minigrid
from gym_minigrid.lib.Direction import Direction
from gym_minigrid.wrappers import *
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.lib.MetricCollector import MetricCollector
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)
def runSteps(env, steps=1, close=True):
    env.render()
    time.sleep(1)
    for i in range(steps):

        obs, reward, done, info = env.step(None)
        if done:
            "Reached the goal"
            break
        env.render()
        time.sleep(1)
    if close:
        env.close()


        
# Load the gym environment
env = gym.make('MultiPedestrian-Empty-5x20-v0')
metricCollector = MetricCollector(env, 0, 100)
agents = []



agent1Position = (3,1)
agent1Speed = 3
agent1 = BlueAdlerPedAgent(
    id=1,
    position=(3,1),
    direction=Direction.East,
    speed=3,
    DML=False,
    p_exchg=0.0
)

agents.append(agent1)

agent2Position = (4,1)
agent2Speed = 3
agent2 = BlueAdlerPedAgent(
    id=2,
    position=(4,1),
    direction=Direction.East,
    speed=3,
    DML=False,
    p_exchg=0.0
)
agents.append(agent2)

# agent2 = BlueAdlerPedAgent(

env.addAgents(agents)

runSteps(env, 1, close=False)
runSteps(env, 2)

env.reset()
env.close()
