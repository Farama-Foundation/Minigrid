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
    env = gym.make('MultiPedestrian-Empty-5x20-v0')  
    
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
def test_2_agents(env):
    # should have a guaranteed lane change. because in current lane gap = 0
    # also how is it moving although gap = 0? if it stayed in current lane, it shouldn't have moved
    # maybe it is calculating gap correctly, but making wrong lane decisions

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
        id=1,
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

    # assert agent1.position == agent1Position
    # assert agent2.position == agent2Position

    runSteps(env, 1)
    # assert agent1.position == agent1Position
    # assert agent2.position == agent2Position

    # assert False


@pytest.mark.caBiAdler
def test_2_agents_for_diagram(env):

    agents = []

    agent1Position = (3,4)
    agent1Speed = 3
    agent1 = BlueAdlerPedAgent(
        id=1,
        position=agent1Position,
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0
    )

    agents.append(agent1)

    # agent2Position = (3,2)
    # agent2Speed = 3
    # agent2 = BlueAdlerPedAgent(
    #     id=1,
    #     position=agent2Position,
    #     direction=Direction.East,
    #     speed=3,
    #     DML=False,
    #     p_exchg=0.0
    # )
    # agents.append(agent2)

    # agent2 = BlueAdlerPedAgent(

    env.addAgents(agents)

    runSteps(env, 1, close=False)

    # assert agent1.position == agent1Position
    # assert agent2.position == agent2Position

    runSteps(env, 1)
    # assert agent1.position == agent1Position
    # assert agent2.position == agent2Position

    # assert False


