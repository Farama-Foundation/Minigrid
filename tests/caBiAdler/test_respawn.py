import logging
import time

import pytest

import gym
import numpy as np

import pedgrid
from pedgrid.agents import BlueAdlerPedAgent
from pedgrid.lib.MetricCollector import MetricCollector
from pedgrid.wrappers import *
from pedgrid.lib.Direction import Direction


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
# def test_2_agents(env):
#     # create two agents facing each other

#     agents = []

#     agent1Position = (1,1)
#     agent1Speed = 3
#     agent1 = BlueAdlerPedAgent(
#         id=1,
#         position=(17,1),
#         direction=Direction.East,
#         speed=3,
#         DML=False,
#         p_exchg=0.0
#     )

#     agents.append(agent1)

#     agent2Position = (2,1)
#     agent2Speed = 3
#     agent2 = BlueAdlerPedAgent(
#         id=1,
#         position=(6,1),
#         direction=Direction.West,
#         speed=3,
#         DML=False,
#         p_exchg=0.0
#     )
#     agents.append(agent2)

#     # agent2 = BlueAdlerPedAgent(

#     env.addPedAgents(agents)

#     runSteps(env, 1, close=False)

#     assert agent1.position == (1, 1)
#     assert agent2.position == (2, 1)

#     runSteps(env, 1)
#     assert agent1.position == agent1Position
#     assert agent2.position == agent2Position

#     # assert False

def test_2_agents(env):
    # create two agents facing each other

    agents = []

    agent1Position = (1,1)
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

    agent2Position = (2,1)
    agent2Speed = 3
    agent2 = BlueAdlerPedAgent(
        id=1,
        position=agent2Position,
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0
    )
    agents.append(agent2)

    # agent2 = BlueAdlerPedAgent(

    env.addPedAgents(agents)

    runSteps(env, 1, close=False)

    assert agent1.position == (1, 1)
    assert agent2.position == (2, 1)

    runSteps(env, 1)
    assert agent1.position == agent1Position
    assert agent2.position == agent2Position

    # assert False

@pytest.mark.caBiAdler
# def test_2_agents_on_finish(env):
#     # create two agents facing each other

#     expectedGap = 1
#     expectedTranslation = expectedGap + 1
#     agents = []

#     agent1Position = (8,1)
#     agent1Speed = 3
#     agent1 = BlueAdlerPedAgent(
#         id=1,
#         position=(18, 1),
#         direction=Direction.East,
#         maxSpeed=agent1Speed,
#         speed=agent1Speed,
#         DML=False,
#         p_exchg=1.0
#     )

#     agents.append(agent1)

#     agent2Position = (1,1)
#     agent2Speed = 3
#     agent2 = BlueAdlerPedAgent(
#         id=1,
#         position=agent2Position,
#         direction=Direction.West,
#         maxSpeed=agent2Speed,
#         speed=agent2Speed,
#         DML=False,
#         p_exchg=1.0
#     )
#     agents.append(agent2)

#     # agent2 = BlueAdlerPedAgent(

#     env.addPedAgents(agents)

#     runSteps(env, 1, close=False)

#     assert agent1.position == (1, 1)
#     assert agent2.position == (18, 1)

#     agent1Position = agent1.position
#     agent2Position = agent2.position

#     runSteps(env, 1)
#     assert agent1.position == (4, 1)
#     assert agent2.position == (15, 1)

def test_2_agents_on_finish(env):
    # create two agents facing each other

    # when agents reach the end of the grid, they should change direction and move back
    
    expectedGap = 1
    expectedTranslation = expectedGap + 1
    agents = []

    agent1Position = (18,1)
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

    agent2Position = (1,1)
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

    env.addPedAgents(agents)

    runSteps(env, 1, close=False)

    assert agent1.position == (18, 1)
    assert agent2.position == (1, 1)

    runSteps(env, 1)
    assert agent1.position == (15, 1)
    assert agent2.position == (4, 1)

def test_2_agents_exchange_on_finish(env):
    # create two agents facing each other

    expectedGap = 2
    expectedTranslation = expectedGap + 1
    agents = []

    agent1Position = (18,1)
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
        direction=Direction.West,
        maxSpeed=agent2Speed,
        speed=agent2Speed,
        DML=False,
        p_exchg=1.0
    )
    agents.append(agent2)

    agent3Position = (1,1)
    agent3Speed = 3
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


    env.addPedAgents(agents)

    runSteps(env, 1, close=False)

    # assert agent1.position == (agent1Position[0] + expectedTranslation, agent1Position[1])
    # assert agent2.position == (agent2Position[0] - expectedTranslation, agent2Position[1])

    agent1Position = agent1.position
    agent2Position = agent2.position

    runSteps(env, 1)
    # assert agent1.position == (agent1Position[0] + agent1Speed, agent1Position[1])
    # assert agent2.position == (agent2Position[0] - agent2Speed, agent2Position[1])



