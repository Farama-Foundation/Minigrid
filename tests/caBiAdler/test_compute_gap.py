import pytest
from gym_minigrid.lib.Direction import Direction
from gym_minigrid.agents import BlueAdlerPedAgent
from gym_minigrid.agents import LaneNum


@pytest.mark.xfail()
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
        position=(3,1),
        direction=Direction.LR,
        speed=3,
        DML=False,
        p_exchg=0.0,
        pedVmax=pedVMax
    )

    agents = [agent1, agent2]

    gap, gap_same, gap_opposite, agentOppIndex = agent1.computeGap(agents, LaneNum.currentLane)

    print(gap, gap_same, gap_opposite, agentOppIndex)

    assert gap == 0
    assert gap_same == 0
    assert gap_opposite == pedVMax
    assert agentOppIndex == -1


