
from pedgrid.lib.Action import Action
from pedgrid.lib.ForwardAction import ForwardAction
from .PedAgent import PedAgent

class TutorialPedAgent(PedAgent):

    """Define a pedestrian that just moves forward in our Second env
    Assignment: make the pedestrian not always keep the same line."""
    
    def parallel1(self, env) -> Action:
        # raise NotImplementedError("parallel1 is not implemented")
        return Action(self, ForwardAction.KEEP)
        # return None

    def parallel2(self, env) -> Action:
        # raise NotImplementedError("parallel2 is not implemented")
        return None