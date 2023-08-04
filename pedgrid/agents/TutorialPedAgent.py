
from pedgrid.lib.Action import Action
from pedgrid.lib.ObjectAction import ObjectAction
from .PedAgent import PedAgent

class TutorialPedAgent(PedAgent):

    """Define a pedestrian that just moves forward in our Second env
    Assignment: make the pedestrian not always keep the same line."""
    
    def parallel1(self, env) -> Action:
        # raise NotImplementedError("parallel1 is not implemented")
        return Action(self, ObjectAction.FORWARD)
        # return None
        #return None
        
        return Action(self, LaneAction.RIGHT)


        #return Action(self, ObjectAction.FORWARD)

    def parallel2(self, env) -> Action:
        # raise NotImplementedError("parallel2 is not implemented")
       # return None
        return Action(self, ObjectAction.FORWARD)
        #return Action(self, ObjectAction.FORWARD)

    