from .PedestrianEnv import PedestrianEnv
from gym_minigrid.register import register

class TwoLaneRoadEnv(PedestrianEnv):
    # Write the outline here how it should work

    # generic object representation
    # generic actors?
    

    pass


register(
    id='TwoLaneRoadEnv-20x80-v0',
    entry_point='gym_minigrid.envs.pedestrian.MultiPedestrianEnv:TwoLaneRoadEnv'
)