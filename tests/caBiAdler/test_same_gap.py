import pytest
from gym_minigrid.lib.Direction import Direction
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.agents import Lanes

def test_same_agents_not_following():
    pedVMax = 4

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(2,1),
        direction=Direction.LR,
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
        direction=Direction.RL,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(4,1),
        direction=Direction.RL,
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
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(4,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    assert sameAgents == [agent2]

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.RL,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(2,1),
        direction=Direction.RL,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    assert sameAgents == [agent2]



@pytest.mark.skip
def test_game_same_0():

    pedVMax = 4

    agent1 = BlueAdlerPedAgent(
        id=1,
        position=(3,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )
    agent2 = BlueAdlerPedAgent(
        id=2,
        position=(4,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    sameAgents, oppAgents = agent1.getSameAndOppositeAgents(agents)

    sameGap = agent1.computeSameGap(sameAgents)

    print(sameGap)
    assert sameGap == 0

