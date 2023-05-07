import logging
from typing import Tuple, List

import numpy as np
import math

from gym_minigrid.agents import LaneNum

from .PedAgent import PedAgent
from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.ForwardAction import ForwardAction
from gym_minigrid.lib.Direction import Direction


class BlueAdlerPedAgent(PedAgent):

    def __init__(
        self, 
        id,
        position: Tuple[int, int], 
        direction: int, # TODO convert direction to enum,
        maxSpeed: float = 4,
        speed: float = 3,
        DML: bool = False, # TODO this is not a property of the agent.
        p_exchg: float = 0.0,
        pedVmax: int = 4
        ):
    
        super().__init__(
            id=id,
            position=position,
            direction=direction,
            maxSpeed=maxSpeed,
            speed=speed,
            DML=DML,
            p_exchg=p_exchg
        )

        self.pedVmax = pedVmax
        # self.speedFixed = True



    # [0] = x axis [1] = y-axis
    # 1 = down face
    # 3 = up face
    def parallel1(self, env): # TODO add type
        """_summary_

        Args:
            agents (_type_): _description_

        Returns:
            _type_: 0 to keep lane, 1 shiftLeft, 2 shiftRight
        """
        # self.speedFixed = False
        agents = env.agents
        #TODO Simulate lane change
        gaps = [None] * 3
        gaps[0] = self.computeGap(agents, LaneNum.currentLane, env)
        if self.canShiftLeft == True:
            gaps[1] = self.computeGap(agents, Lanes.leftLane, env)
        else:
            gaps[1] = -100, 0, 0, None
        if self.canShiftRight == True:
            gaps[2] = self.computeGap(agents, Lanes.rightLane, env)
        else:
            gaps[2] = -100, 0, 0, None
        
        goodLanes = []

        if self.DML and gaps[0][3] is not None: # if agentOppIndex exists, then gapOpp <= 4, prevents default of gapOpp = 0 from messing up code
            gaps[0] = (0, gaps[0][1], gaps[0][2], gaps[0][3]) # set gap = 0
            if gaps[1][1] == 0: # check if left lane gapSame == 0
                goodLanes.append(1)
            if gaps[2][1] == 0: # check if right lane gapSame == 0
                goodLanes.append(2)
        
        # rest of algo
        if self.DML == False or len(goodLanes) == 0:
            maxGap = 0
            for i in range(3):
                maxGap = max(maxGap, gaps[i][0])
            # logging.info(f"id: {self.id} max_gap: {maxGap}")
            for i in range(3):
                if maxGap == gaps[i][0]:
                    goodLanes.append(i)
        # logging.info(f"id: {self.id} goodlanes : {goodLanes}")
        if len(goodLanes) == 1:
            lane = goodLanes[0]
        elif len(goodLanes) == 2:
            if goodLanes[0] == LaneNum.currentLane:
                if np.random.random() > 0.2:
                    lane = goodLanes[0]
                else:
                    lane = goodLanes[1]
            else: #no current lane
                if np.random.random() > 0.5:
                    lane = goodLanes[0]
                else:
                    lane = goodLanes[1]
        else:
            prob = np.random.random()
            if prob > 0.2:
                lane = goodLanes[0]
            elif prob > 0.1:
                lane = goodLanes[1]
            else:
                lane = goodLanes[2]

        # logging.info(f"id: {self.id} lane Chosen: {lane}")
        self.gap = gaps[lane][0]
        self.gapSame = gaps[lane][1]
        self.gapOpp = gaps[lane][2]
        self.closestOpp = gaps[lane][3]

        # logging.info(f"Parallel 1 gap: {self.gap}, gapSame: {self.gapSame}, gapOpp: {self.gapOpp}")

        # return lane
        
        return self.convertLaneDecisionToAction(lane)

    def convertLaneDecisionToAction(self, laneDecision: int) -> LaneAction:
        if laneDecision == 0:
            return None
        if laneDecision == 1:
            return Action(self, LaneAction.LEFT)
        if laneDecision == 2:
            return Action(self, LaneAction.RIGHT)

    def parallel2(self, env): # TODO add type
        # if self.speedFixed == True:
        #     return Action(self, ForwardAction.KEEP)

        agents = env.agents
        self.speed = self.gap
        # if self.gap <= 1 and self.gap == self.gapOpp: # self.gap may have to be 0 instead of 0 or 1
        if self.gap < self.pedVmax and self.gap == self.gapOpp: # self.gap may have to be 0 instead of 0 or 1
        # if self.gap == self.gapOpp: # self.gap may have to be 0 instead of 0 or 1
            if np.random.random() < self.p_exchg:
                self.speed = self.gap + 1
                if self.closestOpp is not None:
                    self.closestOpp.speed = self.gap + 1 # TODO can one agent update another?
            else:
                self.speed = 0
                if self.closestOpp is not None:
                    self.closestOpp.speed = 0

        # logging.info(f"gap: {self.gap}, speed: {self.speed}, gapOpp: {self.gapOpp}")
        
        return Action(self, ForwardAction.KEEP)
        

    def computeGap(self, agents, lane, env=None) -> Tuple[int, int, int, PedAgent]:
        """
        Compute the gap (basically the possible speed ) according to the paper
        """

        laneOffset = 0
        if lane == Lanes.leftLane:
            laneOffset = -1
        elif lane == Lanes.rightLane:
            laneOffset = 1

        sameAgents, oppositeAgents = self.getSameAndOppositeAgents(agents, laneOffset=laneOffset)
       
        gap_same = self.computeSameGap(sameAgents)

        gap_opposite, closestOpp = self.computeOppGapAndAgent(oppositeAgents)


        # if gap > maxSpeed, we only use maxSpeed since we pick lanes in parallel1 with gap size
        # Anything > maxSpeed is irrelevant because it doesn't affect agent movement
        # doesn't affect parallel2 because maxSpeed >= 2 and parallel2 checks for == 0 or <= 1
        gap = min(self.maxSpeed, min(gap_same, gap_opposite))
        # print(f"self position: {self.position}")
        # print(f"id: {self.id} computeGap gap: {gap}, gap_opposite: {gap_opposite}, gap_same: {gap_same}")
        return gap, gap_same, gap_opposite, closestOpp
        
    def getSameAndOppositeAgents(self, agents: List[PedAgent], laneOffset=0) -> Tuple[List[PedAgent], List[PedAgent]]:

        # TODO handle all the corner cases
        opps = []
        sames = []
        for agent2 in agents:

            if agent2 == self:
                continue

            if self.inTheRelevantLane(agent2, laneOffset=laneOffset):
                if self.isFollowing(agent2):
                    sames.append(agent2)
                elif self.isFacing(agent2):
                    opps.append(agent2)

        return sames, opps

    def isFollowing(self, other:PedAgent) -> bool:
        if self.direction != other.direction or self.position == other.position:
            return False
        return self.isBehind(other)

    def isFacing(self, other: PedAgent) -> bool:
        if self.direction == other.direction or self.position == other.position:
            return False
        return self.isBehind(other)

    def isBehind(self, other:PedAgent) -> bool:

        if self.direction == Direction.LR and self.position[0] > other.position[0]:
            return False
        if self.direction == Direction.RL and self.position[0] < other.position[0]:
            return False
        return True

    
    def inTheRelevantLane(self, agent2: PedAgent, laneOffset=0) -> bool:
        return (self.position[1] + laneOffset) == agent2.position[1]

    def distanceTo(self, other:PedAgent) -> int:
        return abs(self.position[0] - other.position[0])
    
    def cellsBetween(self, other: PedAgent) -> int:
        return self.distanceTo(other) - 1

    def computeSameGap(self, sameAgents: List[PedAgent]) -> int:
        gap_same = 2 * self.pedVmax
        for agent2 in sameAgents:
            gap = self.cellsBetween(agent2) # number of cells between?
            if gap < 0:
                print(f"gap: {gap}, {self.position}, {agent2.position}")
                # gap = 0
            assert gap >= 0
            # print("gap: ", gap)

            if gap >= 0 and gap <= 2 * self.pedVmax: # gap must not be negative and less than 8
                gap_same = min(gap_same, gap)

        return gap_same

    def computeOppGapAndAgent(self, opps: List[PedAgent]) -> Tuple[int, PedAgent]:
        """
        Warning, does not check if opps have same direction agents
        returns 1 if the cell between is 1. 
        """
        closestOpp = None
        gap_opposite = self.pedVmax * 2

        for i, agent2 in enumerate(opps):
            
            # gap = self.distanceTo(agent2)
            gap = self.cellsBetween(agent2)
            if gap < 0:
                print(f"gap: {gap}, {self.position}, {agent2.position}")
                # gap = 0
            assert gap >= 0

            gap_agent2 = math.ceil(gap / 2)

            # print("cells between", gap)
            # print("gap_agent2", gap_agent2)

            if gap_agent2 <= gap_opposite:
                closestOpp = agent2
                gap_opposite = gap_agent2

            # if gap >= 0 and gap <= 2 * self.pedVmax: # gap must not be negative and less than 4
            #     if min(gap_opposite, gap // 2) == gap // 2:
            #         closestOpp = agent2
            #     gap_opposite = min(gap_opposite, gap // 2)
        
        return gap_opposite, closestOpp


