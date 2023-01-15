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
def test_2_agents(env):
    # create two agents facing each other

    agents = []

    agent1Position = (3,1)
    agent1Speed = 3
    agent1 = BlueAdlerPedAgent(
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
    agent2 = BlueAdlerPedAgent(
        id=1,
        position=(6,1),
        direction=Direction.RL,
        speed=3,
        DML=False,
        p_exchg=0.0
    )
    agents.append(agent2)

    # agent2 = BlueAdlerPedAgent(

    env.addAgents(agents)

    runSteps(env, 1, close=False)

    assert agent1.position == agent1Position
    assert agent2.position == agent2Position

    runSteps(env, 1)
    assert agent1.position == agent1Position
    assert agent2.position == agent2Position

    assert False



@pytest.mark.caBiAdler
def test_2_agents_exchange(env):
    # create two agents facing each other

    expectedGap = 1
    expectedTranslation = expectedGap + 1
    agents = []

    agent1Position = (8,1)
    agent1Speed = 3
    agent1 = BlueAdlerPedAgent(
        id=1,
        position=agent1Position,
        direction=Direction.LR,
        maxSpeed=agent1Speed,
        speed=agent1Speed,
        DML=False,
        p_exchg=1.0
    )

    agents.append(agent1)

    agent2Position = (11,1)
    agent2Speed = 3
    agent2 = BlueAdlerPedAgent(
        id=1,
        position=agent2Position,
        direction=Direction.RL,
        maxSpeed=agent2Speed,
        speed=agent2Speed,
        DML=False,
        p_exchg=1.0
    )
    agents.append(agent2)

    # agent2 = BlueAdlerPedAgent(

    env.addAgents(agents)

    runSteps(env, 1, close=False)

    assert agent1.position == (agent1Position[0] + expectedTranslation, agent1Position[1])
    assert agent2.position == (agent2Position[0] - expectedTranslation, agent2Position[1])

    agent1Position = agent1.position
    agent2Position = agent2.position

    runSteps(env, 1)
    assert agent1.position == (agent1Position[0] + agent1Speed, agent1Position[1])
    assert agent2.position == (agent2Position[0] - agent2Speed, agent2Position[1])

def test_2_agents_exchange_inthesameplace(env):
    # create two agents facing each other

    expectedGap = 1
    expectedTranslation = expectedGap + 1
    agents = []

    agent1Position = (8,1)
    agent1Speed = 3
    agent1 = BlueAdlerPedAgent(
        id=1,
        position=agent1Position,
        direction=Direction.LR,
        maxSpeed=agent1Speed,
        speed=agent1Speed,
        DML=False,
        p_exchg=1.0
    )

    agents.append(agent1)

    agent2Position = (12,1)
    agent2Speed = 3
    agent2 = BlueAdlerPedAgent(
        id=1,
        position=agent2Position,
        direction=Direction.RL,
        maxSpeed=agent2Speed,
        speed=agent2Speed,
        DML=False,
        p_exchg=1.0
    )
    agents.append(agent2)

    # agent2 = BlueAdlerPedAgent(

    env.addAgents(agents)

    runSteps(env, 1, close=False)

    assert agent1.position == (agent1Position[0] + expectedTranslation, agent1Position[1])
    assert agent2.position == (agent2Position[0] - expectedTranslation, agent2Position[1])

    agent1Position = agent1.position
    agent2Position = agent2.position

    runSteps(env, 1)
    assert agent1.position == (agent1Position[0] + agent1Speed, agent1Position[1])
    assert agent2.position == (agent2Position[0] - agent2Speed, agent2Position[1])
