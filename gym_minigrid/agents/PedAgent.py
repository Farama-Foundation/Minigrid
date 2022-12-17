from gym_minigrid.agents.Agent import Agent
from gym_minigrid.agents import Lanes
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.LaneAction import LaneAction
from gym_minigrid.lib.ForwardAction import ForwardAction
import numpy as np
import logging

class PedAgent(Agent):
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
        agents = env.agents
        #TODO Simulate lane change
        gaps = np.zeros((3, 4)).astype(int)
        gaps[0] = self.computeGap(agents, Lanes.currentLane)
        if self.canShiftLeft == True:
            gaps[1] = self.computeGap(agents, Lanes.leftLane)
        if self.canShiftRight == True:
            gaps[2] = self.computeGap(agents, Lanes.rightLane)
        
        logging.debug('gaps', gaps)
        # confused about DML(Dynamic Multiple Lanes)
        maxGap = 0
        for i in range(3):
            maxGap = max(maxGap, gaps[i][0])
        logging.debug('maxgap', maxGap)
        goodLanes = []
        for i in range(3):
            if maxGap == gaps[i][0]:
                goodLanes.append(i)
        
        if len(goodLanes) == 1:
            lane = goodLanes[0]
        elif len(goodLanes) == 2:
            if goodLanes[0] == Lanes.currentLane:
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

        self.gap = gaps[lane][0]
        self.gapSame = gaps[lane][1]
        self.gapOpp = gaps[lane][2]
        self.agentOppIndex = gaps[lane][3]

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
        agents = env.agents
        self.speed = self.gap
        if self.gapOpp == 0 and self.gap == self.gapOpp: # or <= 1 if using possibly wrong algorithm in paper
            if np.random.random() < self.p_exchg:
                self.speed = self.gap + 1
                agents[self.agentOppIndex].speed = self.gap + 1
        
        return Action(self, ForwardAction.KEEP)
        

    def computeGap(self, agents, lane):
        """
        Compute the gap (basically the possible speed ) according to the paper
        """
        # first sameGap
        postionX = self.position[0]
        postionY = self.position[1]
        gap_opposite = 4
        gap_same = 8
        if lane == Lanes.leftLane:
            postionY -= 1
        elif lane == Lanes.rightLane:
            postionY += 1
        for agent2 in agents:
            #sameGap, so direction must be same and they must be in same lane
            if self.direction != agent2.direction or postionY != agent2.position[1]: 
                continue
            if self.direction == 2: #looking up
                gap = postionX - agent2.position[0]
            elif self.direction == 0: #looking down
                gap = agent2.position[0] - postionY

            if gap > 0 and gap <= 8: # gap must not be negative and less than 8
                gap_same = min(gap_same, gap)

        # now oppGap
        agentOppIndex = -1
        for i, agent2 in enumerate(agents):
            #oppGap, so direction must be different and they must be in same lane
            if self.direction == agent2.direction or postionY != agent2.position[1]: 
                continue
            if self.direction == 2: #looking up
                gap = postionY - agent2.position[0]
            elif self.direction == 0: #looking down
                gap = agent2.position[0] - postionY
            
            if gap > 0 and gap <= 8: # gap must not be negative and less than 4
                if min(gap_opposite, gap/2) == gap/2:
                    agentOppIndex = i
                gap_opposite = min(gap_opposite, gap/2)

        # if gap > maxSpeed, we only use maxSpeed since we pick lanes in parallel1 with gap size
        # Anything > maxSpeed is irrelevant because it doesn't affect agent movement
        # doesn't affect parallel2 because maxSpeed >= 2 and parallel2 checks for == 0 or <= 1
        gap = min(self.maxSpeed, min(gap_same, gap_opposite))
        return gap, gap_same, gap_opposite, agentOppIndex
        

   

from enum import IntEnum

class Lanes(IntEnum):
    currentLane = 0
    leftLane = 1
    rightLane = 2