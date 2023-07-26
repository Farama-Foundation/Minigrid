
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.ForwardAction import ForwardAction
from gym_minigrid.lib.LaneAction import LaneAction
from .PedAgent import PedAgent
import numpy as np

class TutorialPedAgent(PedAgent):

    """Define a pedestrian that just moves forward in our Second env
    Assignment: make the pedestrian not always keep the same line."""
    
    def parallel1(self, env) -> Action:
        # raise NotImplementedError("parallel1 is not implemented")
        return Action(self, ForwardAction.KEEP)
        # return None

    def parallel2(self, env) -> Action:
        # raise NotImplementedError("parallel2 is not implemented")
        return np.random.choice([Action(self, LaneAction.LEFT), Action(self, LaneAction.RIGHT)], (0.5, 0.5))
        return None