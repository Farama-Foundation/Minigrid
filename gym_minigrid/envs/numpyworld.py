from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np


from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np

class NumpyMap(MiniGridEnv):
    """
    Environment created if given a numpy array and index mapping
    """

    def __init__(self):
        super().__init__(grid_size=19, max_steps=100)

    def _gen_grid(self, array, index_mapping):
        # Create the grid
        self.array = array

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                entity_name = index_mapping[array[i][j]]
                for worldobj in WorldObj.__subclasses__():
                    # if entity == worldobj.__name__:
                    #     print('entity')
                    entity = WorldObj.__subclasses__()[array[i][j]]()
                    self.put_obj(entity, i, j)


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info



class NumpyMapFourRooms(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.
    """

    def __init__(self, array, index_mapping):
        self.array = array

        super().__init__(grid_size=41, max_steps=1000)


    def _gen_grid(self, width, height):
        
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create the grid
        for i in range(1, self.array.shape[0]):
            for j in range(1, self.array.shape[1]):
                entity_name = self.index_mapping[self.array[i][j]]
                entity_index = int(self.array[i][j])

                if entity_index != 10 and entity_index != 0:  #'agent' and 'unseen':
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            self.put_obj(entity_class(), j, i)

                elif entity_index == 0:
                	# Place some WorldObj in Unseen, eg. Wall
                    self.put_obj(Wall(), j, i) 

        self.place_agent()
        self.place_obj(Goal())
        self.mission = 'Reach the goal'            


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


class NumpyMapFourRoomsPartialView(NumpyMapFourRooms):
	"""
	Assuming that `grid.npy` exists in the root folder 
	with the approperiate shape for the grid (i.e. 40x40)
	"""
	def __init__(self, numpyFile='numpyworldfiles/map003.npy'):
		self.array = np.load(numpyFile)
		self.index_mapping = {
             0 : 'unseen'        ,
             1 : 'empty'         ,
             2 : 'wall'          ,
             3 : 'floor'         ,
             4 : 'door'          ,
             5 : 'key'           ,
             6 : 'ball'          ,
             7 : 'box'           ,
             8 : 'goal'          ,
             9 : 'lava'          ,
             10: 'agent'          
        }
		super().__init__(self.array, self.index_mapping)



register(
    id='MiniGrid-NumpyMap-v0',
    entry_point='gym_minigrid.envs:NumpyMap'
)
register(
    id='MiniGrid-NumpyMapFourRooms-v0',
    entry_point='gym_minigrid.envs:NumpyMapFourRooms'
)
register(
	
	id='MiniGrid-NumpyMapFourRoomsPartialView-v0',
    entry_point='gym_minigrid.envs:NumpyMapFourRoomsPartialView'
)