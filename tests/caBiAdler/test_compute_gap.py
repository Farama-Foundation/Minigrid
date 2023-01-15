
from gym_minigrid.lib.Direction import Direction
from gym_minigrid.agents import PedAgent
from gym_minigrid.agents import Lanes

def test_game_same_0():

    agent1 = PedAgent(
        id=1,
        position=(3,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0
    )
    agent2 = PedAgent(
        id=2,
        position=(3,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0
    )

    agents = [agent1, agent2]

    gap, gap_same, gap_opposite, agentOppIndex = agent1.computeGap(agents, Lanes.currentLane)

    print(gap, gap_same, gap_opposite, agentOppIndex)

    assert gap == 0
    assert gap_same == 0
    assert gap_opposite == 0
    assert agentOppIndex == -1

def test_game_same_1():

    agent1 = PedAgent(
        id=1,
        position=(3,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0
    )
    agent2 = PedAgent(
        id=2,
        position=(4,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0
    )

    agents = [agent1, agent2]

    gap, gap_same, gap_opposite, agentOppIndex = agent1.computeGap(agents, Lanes.currentLane)

    print(gap, gap_same, gap_opposite, agentOppIndex)

    assert gap == 0
    assert gap_same == 0
    assert gap_opposite == 0
    assert agentOppIndex == -1


def test_game_same_1_1():

    agent1 = PedAgent(
        id=1,
        position=(3,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0
    )
    agent2 = PedAgent(
        id=2,
        position=(2,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0
    )

    agents = [agent1, agent2]

    gap, gap_same, gap_opposite, agentOppIndex = agent1.computeGap(agents, Lanes.currentLane)

    print(gap, gap_same, gap_opposite, agentOppIndex)

    assert gap == 0
    assert gap_same == 0
    assert gap_opposite == 0
    assert agentOppIndex == -1