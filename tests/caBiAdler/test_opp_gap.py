import pytest
from gym_minigrid.lib.Direction import Direction
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.agents import Lanes





def test_gap_facing_0():

    pedVMax = 4

    y = 1
    x = 30

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(x,y),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(x + 1,1),
        direction=Direction.RL,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    gap_opposite, agentOppIndex = agent1.computeOppGapAndIndex(oppAgents)

    print(gap_opposite, agentOppIndex)

    assert oppAgents == [agent2]

    assert gap_opposite == 0
    # assert agentOppIndex == -1
