import pytest
from gym_minigrid.lib.Direction import Direction
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.agents import LaneNum
import math




def test_gap_facing_0():

    pedVMax = 4

    y = 1
    x = 30

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(x,y),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(x + 1,1),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    gap_opposite, closestOpp = agent1.computeOppGapAndAgent(oppAgents)

    print(gap_opposite, closestOpp)

    assert oppAgents == [agent2]
    assert closestOpp == agent2

    assert gap_opposite == 0
    # assert agentOppIndex == -1

@pytest.mark.xfail 
def test_gap_facing_inthesameplace_and_gap_is_negative(): # should be impossible

    pedVMax = 4

    y = 1
    x = 30

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(x,y),
        direction=Direction.East,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(x,y),
        direction=Direction.West,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    gap_opposite, closestOpp = agent1.computeOppGapAndAgent(oppAgents)

    print(gap_opposite, closestOpp)

    assert oppAgents == []
    assert closestOpp == None

    assert gap_opposite >= 0
    # assert agentOppIndex == -1


def test_gp_opp_0_to_10_LR():
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

    for cellsBetween in range(10):
        agent2 = BlueAdlerPedAgent(
            id=2,
            position=(x+cellsBetween+1,y),
            direction=Direction.West,
            speed=3,
            DML=False,
            p_exchg=0.0,
            pedVmax=pedVMax
        )

        agents = [agent1, agent2]

        sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents, laneOffset=0)
        gap_opposite, closestOpp = agent1.computeOppGapAndAgent(oppAgents)

        expectedGap = min(pedVMax,  math.ceil(cellsBetween / 2))
        assert oppAgents == [agent2]
        assert gap_opposite == expectedGap
        if math.ceil(cellsBetween / 2) <= pedVMax:
            assert closestOpp == agent2
        else:
            assert closestOpp is None


def test_gp_opp_0_to_10_RL():
    pedVMax = 4
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

    for cellsBetween in range(10):
        agent2 = BlueAdlerPedAgent(
            id=2,
            position=(x - cellsBetween - 1,y),
            direction=Direction.East,
            speed=3,
            DML=False,
            p_exchg=0.0,
            pedVmax=pedVMax
        )

        agents = [agent1, agent2]

        sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents, laneOffset=0)
        gap_opposite, closestOpp = agent1.computeOppGapAndAgent(oppAgents)

        expectedGap = min(pedVMax,  math.ceil(cellsBetween / 2))
        assert oppAgents == [agent2]
        assert gap_opposite == expectedGap
        if math.ceil(cellsBetween / 2) <= pedVMax:
            assert closestOpp == agent2
        else:
            assert closestOpp is None