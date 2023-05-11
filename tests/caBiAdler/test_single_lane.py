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
        direction=Direction.East,
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
        direction=Direction.West,
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
        direction=Direction.East,
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
        direction=Direction.West,
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

    expectedGap = 2
    expectedTranslation = expectedGap + 1
    agents = []

    agent1Position = (8,1)
    agent1Speed = 3
    agent1 = BlueAdlerPedAgent(
        id=1,
        position=agent1Position,
        direction=Direction.East,
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
        direction=Direction.West,
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

def test_3_agents_stuck(env):
    # create two agents facing each other

    expectedGap = 2
    expectedTranslation = expectedGap + 1
    agents = []

    agent1Position = (1,1)
    agent1Speed = 3
    agent1 = BlueAdlerPedAgent(
        id=1,
        position=agent1Position,
        direction=Direction.East,
        maxSpeed=agent1Speed,
        speed=agent1Speed,
        DML=False,
        p_exchg=1.0
    )

    agents.append(agent1)

    agent2Position = (2,1)
    agent2Speed = 3
    agent2 = BlueAdlerPedAgent(
        id=1,
        position=agent2Position,
        direction=Direction.East,
        maxSpeed=agent2Speed,
        speed=agent2Speed,
        DML=False,
        p_exchg=1.0
    )
    agents.append(agent2)

    agent3Position = (3,1)
    agent3Speed = 1
    agent3 = BlueAdlerPedAgent(
        id=1,
        position=agent3Position,
        direction=Direction.East,
        maxSpeed=agent3Speed,
        speed=agent3Speed,
        DML=False,
        p_exchg=1.0
    )
    agents.append(agent3)
    # agent2 = BlueAdlerPedAgent(

    env.addAgents(agents)

    runSteps(env, 1, close=False)

    # assert agent1.position == (agent1Position[0] + expectedTranslation, agent1Position[1])
    # assert agent2.position == (agent2Position[0] - expectedTranslation, agent2Position[1])

    agent1Position = agent1.position
    agent2Position = agent2.position

    runSteps(env, 3)
    # assert agent1.position == (agent1Position[0] + agent1Speed, agent1Position[1])
    # assert agent2.position == (agent2Position[0] - agent2Speed, agent2Position[1])


# opposite gap issues
# if gap = floor(cellsBetween / 2)
# case cells = 0, gap = 0, speed=1, they do not overlap
# cells = 1, gap = 0, speed = 1 they do not overlap
# cells = 2, gap = 1, speed = 2, they do not overlap
# cells = 3, gap = 1, speed = 2 they do not overlap
# cells = 4, gap = 2, speed = 3 they do not overlap
# cells = 5, gap = 2, speed = 3 they do not overlap
