from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pickle
from pathlib import Path
import os
from copy import deepcopy, copy

from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.DungeonRooms import DungeonRooms
from mazelib.solve.ShortestPaths import ShortestPaths
from mazelib.solve.ShortestPath import ShortestPath
from mazelib.solve.BacktrackingSolver import BacktrackingSolver
from mazelib.solve.Chain import Chain
from mazelib.solve.RandomMouse import RandomMouse
from gym_minigrid.envs.multiroom_mod import MultiRoomEnv
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.minigrid import Grid as Minigrid_Grid
from gym_minigrid.minigrid import OBJECT_TO_IDX as Minigrid_OBJECT_TO_IDX
from gym_minigrid.minigrid import IDX_TO_OBJECT as Minigrid_IDX_TO_OBJECT

# Map of object type to integers
OBJECT_TO_CHANNEL_AND_IDX = {
    'empty'         : (0, 0),
    'wall'          : (0, 1),
    'goal'          : (2, 1),
    'agent'         : (1, 1),
    'start'         : (1, 1),
}


class GridNavDatasetGenerator:

    def __init__(self, dataset_meta: Dict[str, Any], batches_meta: List[Dict[str, Any]], save_dir: str):
        self.dataset_meta = dataset_meta
        self.batches_meta = batches_meta
        self.base_dir = str(Path(__file__).resolve().parent) + '/datasets/'
        self.save_dir = self.base_dir + save_dir + '/'
        self.generated_batches = []
        self.generated_labels = []
        self.generated_label_contents = []

        self.feature_shape = (self.dataset_meta['data_dim'][0], self.dataset_meta['data_dim'][1], 3)

    def generate_dataset(self, normalise_difficulty: bool = True):

        for i, batch_meta in enumerate(self.batches_meta):
            batch_g = BatchGenerator(batch_meta, self.dataset_meta)
            batch_features, batch_label_ids, batch_label_contents = batch_g.generate_batch()
            self.generated_batches.append(batch_features)
            self.generated_labels.append(batch_label_ids)
            self.generated_label_contents.append(batch_label_contents)

        if normalise_difficulty: self.normalise_difficulty()

        # creates folder if it does not exist.
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for i in range(len(self.generated_batches)):
            self.save_batch(self.generated_batches[i], self.generated_labels[i],
                            self.generated_label_contents[i], self.batches_meta[i])

        self.save_dataset_meta()

    def save_batch(self, batch_data: np.ndarray, batch_labels: np.ndarray,
                   batch_label_contents: Dict[int, Any], batch_meta: Dict[str, Any]):

        entry = {'data': batch_data, 'labels': batch_labels,
                 'label_contents': batch_label_contents, 'batch_meta': batch_meta}

        filename = self.save_dir + batch_meta['output_file']
        with open(filename, 'wb') as f:
            pickle.dump(entry, f)

    def save_dataset_meta(self):

        entry = self.dataset_meta

        filename = self.save_dir + self.dataset_meta['output_file']
        with open(filename, 'wb') as f:
            pickle.dump(entry, f)

    def normalise_difficulty(self):
        pass  # TODO: implement


class BatchGenerator:

    def __new__(cls, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any]):

        if batch_meta['task_structure'] == 'maze':
            instance = super().__new__(MazeBatch)
        elif batch_meta['task_structure'] == 'rooms_unstructured_layout':
            instance = super().__new__(RoomsUnstructuredBatch)
        else:
            raise KeyError("Task Structure was not recognised")

        instance.__init__(batch_meta, dataset_meta)

        return instance

    def __init__(self):
        raise RuntimeError(f"{type(self)} is a Class Factory. Assign it to a variable. ")


class Batch:

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any]):
        self.batch_meta = batch_meta
        self.dataset_meta = dataset_meta
        self.data_type = dataset_meta['data_type']

    def generate_batch(self):

        features = self.generate_data()
        if self.data_type == 'gridworld':
            pass
        elif self.data_type == 'grid':
            features = self.encode_gridworld_to_grid(features)
        else:
            raise KeyError(f"Data Type '{self.data_type}' not recognised")
        solutions = self.generate_solutions(features)
        label_ids, label_content = self.generate_labels(solutions)

        return features, label_ids, label_content

    def generate_data(self):
        raise NotImplementedError

    def generate_labels(self, solutions: List[List[Tuple]]) -> Tuple[
        np.ndarray, Dict[int, Any]]:
        # call a specific quantity with
        # labels[dataset_meta['label_descriptors'].index('wanted_label_descriptor')][label_id]

        batch_id = [self.batch_meta['batch_id']] * self.batch_meta['batch_size']  # from batch_meta

        task_difficulty = self.task_difficulty(solutions, self.dataset_meta['difficulty_descriptors'])

        seed = [0] * self.batch_meta['batch_size']  # TODO: implement

        # task structure one-hot vector [0,1]
        task_structure = np.zeros((self.batch_meta['batch_size'], len(self.dataset_meta['task_structure_descriptors'])),
                                  dtype=int)
        batch_task_structure_idx = self.dataset_meta['task_structure_descriptors'].index(self.batch_meta['task_structure'])
        task_structure[:, batch_task_structure_idx] = 1

        # TODO: make robust against change of label descriptors (or declare label descriptors as a class var)
        # TODO: assert checks that they are all the right dimensions
        label_ids = np.arange(self.batch_meta['batch_size'])
        label_contents = {0: task_difficulty, 1: task_structure, 2: batch_id, 3: seed, 4: solutions}

        return label_ids, label_contents

    #TODO implement
    @staticmethod
    def generate_solutions(features) -> List[List[Tuple]]:

        solutions = []
        for layout in features:
            optimal_trajectory = (0,0)
            solutions.append(optimal_trajectory)

        return solutions

    @staticmethod
    def task_difficulty(solutions: List[List[Tuple]], difficulty_descriptors: List[str]) -> np.ndarray:

        difficulty_metrics = np.zeros((len(solutions), len(difficulty_descriptors)))

        if 'shortest_path' in difficulty_descriptors:
            shortest_path_ind = difficulty_descriptors.index('shortest_path')
            shortest_path_length = [len(solutions[i]) for i in range(len(solutions))]
            difficulty_metrics[:, shortest_path_ind] = shortest_path_length

        # TODO: add other difficulty metrics

        return difficulty_metrics

    @staticmethod
    def encode_maze_to_gridworld(mazes: Union[Maze, List[Maze]]) -> np.ndarray:

        if isinstance(mazes, Maze):
            mazes = [mazes]

        # Obtain the different channels
        grids = np.array([mazes[i].grid for i in range(len(mazes))])
        start_positions_indices = np.array([[i, mazes[i].start[0], mazes[i].start[1]] for i in range(len(mazes))])
        goal_positions_indices = np.array([[i, mazes[i].end[0], mazes[i].end[1]] for i in range(len(mazes))])
        start_position_channels, goal_position_channels = (np.zeros(grids.shape) for i in range(2))
        start_position_channels[tuple(start_positions_indices.T)] = 1
        goal_position_channels[tuple(goal_positions_indices.T)] = 1

        # merge
        features = np.stack((grids, start_position_channels, goal_position_channels), axis=-1)

        return features

    @staticmethod
    def encode_minigrid_to_gridworld(envs: List[MiniGridEnv]) -> np.ndarray:
        minigrid_grid_arrays = [env.grid.encode()[:, :, 0] for env in envs]

        for i in range(len(envs)):
            minigrid_grid_arrays[i][tuple(envs[i].agent_pos)] = Minigrid_OBJECT_TO_IDX['agent']
        minigrid_grid_arrays = np.array(minigrid_grid_arrays)

        objects_idx = np.unique(minigrid_grid_arrays)
        object_instances = [Minigrid_IDX_TO_OBJECT[obj] for obj in objects_idx]

        features = np.zeros((*minigrid_grid_arrays.shape, 3))
        for obj in object_instances:
            try:
                c = OBJECT_TO_CHANNEL_AND_IDX[obj][0]
                mini_v = Minigrid_OBJECT_TO_IDX[obj]
                v = OBJECT_TO_CHANNEL_AND_IDX[obj][1]
                features[..., c] = np.where(minigrid_grid_arrays == mini_v, v, features[..., c])
            except KeyError:
                raise KeyError("Mismatch between Minigrid generated objects and admissible objects for dataset.")

        return features

    @staticmethod
    def encode_gridworld_to_maze(grids: np.ndarray) -> List[Maze]:
        # Set up maze generator
        mazes = [Maze() for i in range(grids.shape[0])]  # TODO: here add seed argument later
        for (maze, grid) in zip(mazes, grids):
            maze.grid = np.int8(grid[...,0])
            maze.start = tuple(np.argwhere(grid[..., 1] == 1)[0])
            maze.end = tuple(np.argwhere(grid[..., 2] == 1)[0])

        return mazes

    @staticmethod
    def encode_gridworld_to_minigrid(mazes: np.ndarray, config: Dict) -> List[Any]:
        raise NotImplementedError

    @staticmethod
    def encode_gridworld_to_grid(gridworlds: np.ndarray):
        # gridworls shape: [m, odd, odd, 3]
        assert gridworlds.shape[1] % 2 == 1 and gridworlds.shape[2] % 2 == 1
        grid_layout_dim = (gridworlds.shape[0], int(np.floor(gridworlds.shape[1]/2)), int(np.floor(gridworlds.shape[2]/2)), 2)
        grid_layouts = np.zeros(grid_layout_dim)

        layout_channel = OBJECT_TO_CHANNEL_AND_IDX['empty'][0]
        empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]

        for m in range(grid_layouts.shape[0]):
            for i in range(grid_layouts.shape[1]):
                for j in range(grid_layouts.shape[2]):
                    ind_gridworld = (m, int(i*2+1), int(j*2+1), layout_channel)
                    ind_gridworld_right = list(ind_gridworld)
                    ind_gridworld_right[2] += 1
                    ind_gridworld_bot = list(ind_gridworld)
                    ind_gridworld_bot[1] += 1
                    if gridworlds[ind_gridworld] == empty_idx:
                        if gridworlds[tuple(ind_gridworld_right)] == empty_idx:
                            grid_layouts[m,i,j,0] = 1
                        if gridworlds[tuple(ind_gridworld_bot)] == empty_idx:
                            grid_layouts[m, i, j, 1] = 1

        #TODO: use object dictionary
        start_channels, goal_channels = (np.zeros(grid_layout_dim[:-1]) for i in range(2))
        start_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['start'][0]]
                                        == OBJECT_TO_CHANNEL_AND_IDX['start'][1])
        start_inds_grid = (start_inds_gridworld[0], np.floor(start_inds_gridworld[1] / 2).astype(int),
                           np.floor(start_inds_gridworld[2] / 2).astype(int))
        goal_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['goal'][0]])
        goal_inds_grid = (goal_inds_gridworld[0], np.floor(goal_inds_gridworld[1] / 2).astype(int),
                           np.floor(goal_inds_gridworld[2] / 2).astype(int))

        #TODO use object dictionary
        start_channels[start_inds_grid] = 1
        goal_channels[goal_inds_grid] = 1

        # merge
        grids = np.stack((grid_layouts[...,0], grid_layouts[...,1], start_channels, goal_channels), axis=-1)
        return grids

    @staticmethod
    def encode_grid_to_gridworld():
        raise NotImplementedError

    @staticmethod
    def encode_grid_to_graph():
        raise NotImplementedError

    #Note: will need extra args for a deterministic transformation
    @staticmethod
    def encode_graph_to_grid():
        raise NotImplementedError


class MazeBatch(Batch):

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any]):
        super().__init__(batch_meta, dataset_meta)

    def generate_data(self):
        # Set up maze generator
        maze_generator = Maze()  # TODO: here add seed argument later
        maze_size_arg = [int((x - 1) / 2) for x in self.dataset_meta['data_dim']] #TODO! change

        # Set up generating algorithm
        if self.batch_meta['generating_algorithm'] == 'Prims':
            maze_generator.generator = Prims(*maze_size_arg)
        else:
            raise KeyError("Maze generating algorithm was not recognised")

        batch_features = []
        for i in range(self.batch_meta['batch_size']):
            # maze_generator.set_seed(self.batch_seeds[i]) #TODO
            maze_generator.generate()
            maze_generator.generate_entrances(False, False)
            features = self.encode_maze_to_gridworld(maze_generator)
            batch_features.append(features)

        batch_features = np.squeeze(batch_features)
        return batch_features

    # def generate_solutions(self, features) -> List[List[Tuple]]:
    #
    #     solutions = []
    #     # Set up solving algorithm
    #     if self.batch_meta['solving_algorithm'] == 'ShortestPaths':
    #         maze_generator.solver = ShortestPaths()
    #     else:
    #         raise KeyError("Maze solving algorithm was not recognised")
    #
    #     for i in range(self.batch_meta['batch_size']):
    #         maze_generator.solve()
    #         optimal_trajectory = maze_generator.solutions[0]  # TODO Check this is the right one.
    #         solutions.append(optimal_trajectory)


class RoomsUnstructuredBatch(Batch):

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any]):
        super().__init__(batch_meta, dataset_meta)

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, Any]]:
        # Set up generator
        #TODO do seed properly
        envs = [MultiRoomEnv(minNumRooms=4, maxNumRooms=12, maxRoomSize=10, grid_size=self.dataset_meta['data_dim'][0],
                             seed=np.random.randint(100000)) for i in range(self.batch_meta['batch_size'])]

        batch_features = self.encode_minigrid_to_gridworld(envs)
        mazes = self.encode_gridworld_to_maze(batch_features)

        return batch_features

    #TODO: Fix
    # def generate_solutions(self, features) -> List[List[Tuple]]:
    #
    #     solutions = []
    #     for layout in features:
    #         optimal_trajectory = (0,0)  # TODO Check this is the right one.
    #         solutions.append(optimal_trajectory)
    #
    #     return solutions


if __name__ == '__main__':
    dataset_meta = {
        'output_file': 'dataset.meta',
        'data_type': 'grid', #types: gridworld, grid, graph
        'data_dim': (27, 27),  # TODO: assert odd if data_type is 'gridworld'
        'task_type': 'find_goal',
        'label_descriptors': [
            'difficulty_metrics',
            'task structure',
            'batch_id',
            'seed',
            'optimal_trajectory',
        ],
        'difficulty_descriptors': [
            'shortest_path',
            'full_exploration',
        ],
        'task_structure_descriptors': [
            'rooms_unstructured_layout',
            'rooms_square_layout',
            'maze',
            'dungeon',
        ],
        'feature_descriptors': [
            'walls',
            'start_position',
            'goal_position',
        ],
        'generating_algorithms_descriptors': [
            'Prims',
        ],
        'solving_algorithms_descriptors': [
            'ShortestPaths',
        ],
    }

    batches_meta = [
        {
            'output_file': 'batch_0.data',
            'batch_size': 10,
            'batch_id': 0,
            'task_structure': 'maze',
            'generating_algorithm': 'Prims',
            'generating_algorithm_options': [

            ],
            'solving_algorithm': 'ShortestPaths',
            'solving_algorithm_options': [

            ],
        },
    ]

    # batches_meta = [
    #     {
    #         'output_file': 'batch_1.data',
    #         'batch_size': 10,
    #         'batch_id': 0,
    #         'task_structure': 'rooms_unstructured_layout',
    #         'generating_algorithm': 'Minigrid_MultiRoom',
    #         'generating_algorithm_options': [
    #
    #         ],
    #         'solving_algorithm': 'ShortestPaths',
    #         'solving_algorithm_options': [
    #
    #         ],
    #     },
    #     {
    #         'output_file': 'batch_2.data',
    #         'batch_size': 10,
    #         'batch_id': 0,
    #         'task_structure': 'rooms_unstructured_layout',
    #         'generating_algorithm': 'Minigrid_MultiRoom',
    #         'generating_algorithm_options': [
    #
    #         ],
    #         'solving_algorithm': 'ShortestPaths',
    #         'solving_algorithm_options': [
    #
    #         ],
    #     },
    #     {
    #         'output_file': 'batch_3.data',
    #         'batch_size': 10,
    #         'batch_id': 0,
    #         'task_structure': 'rooms_unstructured_layout',
    #         'generating_algorithm': 'Minigrid_MultiRoom',
    #         'generating_algorithm_options': [
    #
    #         ],
    #         'solving_algorithm': 'ShortestPaths',
    #         'solving_algorithm_options': [
    #
    #         ],
    #     },
    #     {
    #         'output_file': 'test_batch.data',
    #         'batch_size': 10,
    #         'batch_id': 0,
    #         'task_structure': 'rooms_unstructured_layout',
    #         'generating_algorithm': 'Minigrid_MultiRoom',
    #         'generating_algorithm_options': [
    #
    #         ],
    #         'solving_algorithm': 'ShortestPaths',
    #         'solving_algorithm_options': [
    #
    #         ],
    #     },
    # ]

    dataset_directory = 'test' + str(batches_meta[0]['batch_size']) + 'x' + str(dataset_meta['data_dim'][0])
    MazeGenerator = GridNavDatasetGenerator(dataset_meta=dataset_meta, batches_meta=batches_meta, save_dir=dataset_directory)
    MazeGenerator.generate_dataset()

    print("Done")
