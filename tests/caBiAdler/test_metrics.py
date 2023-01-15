import logging
import time

import pytest

import gym
import numpy as np

import gym_minigrid
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.lib.MetricCollector import MetricCollector
from gym_minigrid.wrappers import *
from gym_minigrid.lib.Direction import Direction


logging.basicConfig(level=logging.INFO)

@pytest.fixture
def env():
    env = gym.make('MultiPedestrian-Empty-1x20-v0')  
    env.reset()
    return env


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


@pytest.mark.caBiAdler
def test_2_agents_speed(env):
    # create two agents facing each other
    logging.basicConfig(level=logging.INFO)
    metricCollector = MetricCollector(env)
    agents = []

    agent1Position = (3,1)
    agent1Speed = 3
    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.LR,
        speed=4,
        DML=False,
        p_exchg=0.0
    )

    agents.append(agent1)

    agent2Position = (10,1)
    agent2Speed = 2
    agent2 = BlueAdlerPedAgent(
        id=1,
        position=(14,1),
        direction=Direction.RL,
        speed=2,
        DML=False,
        p_exchg=0.0
    )
    agents.append(agent2)


    env.addAgents(agents)

 
    runSteps(env, 20)



    # stepStats = metricCollector.getStatistics()[0]
    # avgSpeed = sum(stepStats["xSpeed"]) / len(stepStats["xSpeed"])
    # logging.info("Average speed: " + str(avgSpeed))
    # volumeStats = metricCollector.getStatistics()[1]
    # avgVolume = sum(volumeStats) / len(volumeStats)
    # logging.info("Average volume: " + str(avgVolume))

