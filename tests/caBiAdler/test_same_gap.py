import pytest
from pedgrid.lib.Direction import Direction
from pedgrid.agents import BlueAdlerPedAgent
from pedgrid.agents import LaneNum

def test_same_agents_not_following():
    pedVMax = 4

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(2,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    assert sameAgents == []

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(4,1),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    assert sameAgents == []


def test_same_agents_other_lane_no_offset():
    pedVMax = 4

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(4,2),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    assert sameAgents == []

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(4,2),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    assert sameAgents == []

def test_same_agents_1_following():
    pedVMax = 4

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(4,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)
    gap = agent1.computeSameGap(sameAgents)

    assert sameAgents == [agent2]
    assert gap == 0

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(2,1),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)
    gap = agent1.computeSameGap(sameAgents)


    assert sameAgents == [agent2]
    assert gap == 0

def test_same_agents_1_following_other_lane():
    pedVMax = 4

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(4,2),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents, laneOffset=1)
    gap = agent1.computeSameGap(sameAgents)


    assert sameAgents == [agent2]
    assert gap == 0

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(2,0),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents, laneOffset=-1)
    gap = agent1.computeSameGap(sameAgents)
    assert sameAgents == [agent2]
    assert gap == 0



def test_game_same_2():

    pedVMax = 4

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(5,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    
    agent3 = BlueAdlerPedAgent(
        id=2,
        position=(7,1),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    
    # opp
    oppAgent1 = BlueAdlerPedAgent(
        id=2,
        position=(5,1),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    # other lane
    
    otherLaneAgent1 = BlueAdlerPedAgent(
        id=2,
        position=(4,2),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2, agent3, oppAgent1, otherLaneAgent1]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)
    gap = agent1.computeSameGap(sameAgents)

    assert sameAgents == [agent2, agent3]
    assert oppAgents == [oppAgent1]
    assert gap == 1





def test_game_same_0_to_10_LR():
    pedVMax = 3
    y = 1
    x = 3

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(x,y),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    for expectedGap in range(10):
        agent2 = BlueAdlerPedAgent(
            id=2,
            position=(x+expectedGap+1,y),
            direction=Direction.East,
            speed=3,
            DML=False,
            p_exchg=0.0,
            pedVmax=pedVMax
        )

        agents = [agent1, agent2]

        sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents, laneOffset=0)
        gap = agent1.computeSameGap(sameAgents)

        expectedGap = min(pedVMax*2, expectedGap)
        assert sameAgents == [agent2]
        assert gap == expectedGap


def test_game_same_0_to_10_RL():
    pedVMax = 3
    y = 1
    x = 30

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(x,y),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    for expectedGap in range(10):
        agent2 = BlueAdlerPedAgent(
            id=2,
            position=(x - expectedGap - 1,y),
            direction=Direction.West,
            speed=3,
            DML=False,
            p_exchg=0.0,
            pedVmax=pedVMax
        )

        agents = [agent1, agent2]

        sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents, laneOffset=0)
        gap = agent1.computeSameGap(sameAgents)

        expectedGap = min(pedVMax*2, expectedGap)
        assert sameAgents == [agent2]
        assert gap == expectedGap