import logging
import time

import pytest

import gym
import numpy as np

import gym_minigrid
from gym_minigrid.agents import PedAgent
from gym_minigrid.lib.MetricCollector import MetricCollector
from gym_minigrid.wrappers import *
from gym_minigrid.lib.Direction import Direction


logging.basicConfig(level=logging.INFO)

@pytest.fixture
def env():
    return gym.make('MultiPedestrian-Empty-1x20-v0')    


def runSteps(env, steps=1, close=True):
    env.reset()
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
def test_2_agents(env):
    # create two agents facing each other

    agents = []

    agent1Position = (3,1)
    agent1Speed = 3
    agent1 = PedAgent(
        id=1,
        position=(3,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0
    )

    agents.append(agent1)

    agent2Position = (6,1)
    agent2Speed = 3
    agent2 = PedAgent(
        id=1,
        position=(6,1),
        direction=Direction.RL,
        speed=3,
        DML=False,
        p_exchg=0.0
    )
    agents.append(agent2)

    # agent2 = PedAgent(

    env.addAgents(agents)

    runSteps(env, 1, close=False)

    assert agent1.position == agent1Position
    assert agent2.position == agent2Position

    runSteps(env, 2)
    assert agent1.position == agent1Position
    assert agent2.position == agent2Position