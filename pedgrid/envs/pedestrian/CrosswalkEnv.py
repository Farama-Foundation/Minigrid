from pedgrid.agents.Agent import Agent
from .PedestrianEnv import MultiPedestrianEnv
import typing as List

class CrosswalkEnv(MultiPedestrianEnv):

    # refactor if we have time (add Crosswalk object)
    
    def __init__(
        self,
        agents: List[Agent]=None,
        width=8,
        height=8,
        crosswalkPosition=None,
        crosswalkOrigin=None,
        crosswalkSize = 3
    ):

        super().__init__(agents, width, height)

        if crosswalkPosition is None:
            crosswalkPosition = (0, (height // 2) - (crosswalkSize // 2))
        
        if crosswalkOrigin is None:
            crosswalkOrigin = (width // 2, crosswalkPosition[1] + (crosswalkSize // 2))

        self.crosswalkOrigin = crosswalkOrigin
        self.crosswalkPosition = crosswalkPosition
        self.crosswalkSize = crosswalkSize

    def getCrossWalkPoint(self, worldPoint):
        return (worldPoint[0] - self.crosswalkOrigin[0], worldPoint[1] - self.crosswalkOrigin[1])

        