import gym_minigrid
from gym_minigrid.lib.Action import Action
from gym_minigrid.lib.ObjectAction import ObjectAction
from gym_minigrid.lib.LaneAction import LaneAction
from .PedAgent import PedAgent

class TutorialPedAgent(PedAgent):

    """Define a pedestrian that just moves forward in our Second env
    Assignment: make the pedestrian not always keep the same line."""
    
    def parallel1(self, env) -> Action:
        # raise NotImplementedError("parallel1 is not implemented")
        return Action(self, ForwardAction.KEEP)
        # return None
        #return None
        
        return Action(self, LaneAction.RIGHT)


        #return Action(self, ObjectAction.FORWARD)

    def parallel2(self, env) -> Action:
        # raise NotImplementedError("parallel2 is not implemented")
       # return None
        return Action(self, ObjectAction.FORWARD)
        #return Action(self, ObjectAction.FORWARD)

    