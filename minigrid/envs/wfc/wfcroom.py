from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pickle
from pathlib import Path
import os

from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.DungeonRooms import DungeonRooms
from mazelib.solve.ShortestPaths import ShortestPaths


class MazeDatasetGenerator:

    def __init__(self, dataset_meta: Dict[str, Any], batches_meta: List[Dict[str, Any]], save_dir: str):
        self.dataset_meta = dataset_meta
        self.batches_meta = batches_meta
        self.base_dir = str(Path(__file__).resolve().parent) + '/datasets/'
        self.save_dir = self.base_dir + save_dir + '/'
        self.generated_batches = []
        self.generated_labels = []
        self.generated_label_contents = []

        self.feature_shape = (self.dataset_meta['maze_size'][0] * self.dataset_meta['maze_size'][1], 3)

    def generate_data(self, normalise_difficulty: bool = True):

        for i, batch_meta in enumerate(self.batches_meta):
            batch_features, batch_labels, batch_label_contents = self.generate_batch(batch_meta)
            self.generated_batches.append(batch_features)
            self.generated_labels.append(batch_labels)
            self.generated_label_contents.append(batch_label_contents)

        if normalise_difficulty: self.normalise_difficulty()

        # creates folder if it does not exist.
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for i in range(len(self.generated_batches)):
            self.save_batch(self.generated_batches[i], self.generated_labels[i],
                            self.generated_label_contents[i], self.batches_meta[i])

        self.save_dataset_meta()

    def generate_batch(self, batch_meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[int, Any]]:

        # Set up maze generator
        maze_generator = Maze()  # here add seed argument later
        maze_size_arg = [int((x - 1) / 2) for x in self.dataset_meta['maze_size']]

        # Set up generating algorithm
        if batch_meta['generating_algorithm'] == 'Prims':
            maze_generator.generator = Prims(maze_size_arg[0], maze_size_arg[1])  # TODO try *maze_size_arg
        else:
            raise KeyError("Maze generating algorithm was not recognised")

        # Set up solving algorithm
        if batch_meta['solving_algorithm'] == 'ShortestPaths':
            maze_generator.solver = ShortestPaths()
        else:
            raise KeyError("Maze solving algorithm was not recognised")

        batch_features = []
        solutions = []
        for i in range(batch_meta['batch_size']):
            # maze_generator.set_seed(self.batch_seeds[i]) #TODO
            maze_generator.generate()
            maze_generator.generate_entrances(False, False)
            maze_generator.solve()
            features = self.encode_mazes(maze_generator)
            optimal_trajectory = maze_generator.solutions[0]  # TODO Check this is the right one.
            batch_features.append(features)
            solutions.append(optimal_trajectory)

        batch_labels, batch_label_contents = self.generate_labels(solutions, batch_meta)
        batch_features = np.squeeze(batch_features)
        return batch_features, batch_labels, batch_label_contents

    def generate_labels(self, solutions: List[List[Tuple]], batch_meta: Dict[str, Any]) -> Tuple[
        np.ndarray, Dict[int, Any]]:
        # call a specific quantity with
        # labels[dataset_meta['label_descriptors'].index('wanted_label_descriptor')][label_id]

        batch_id = [batch_meta['batch_id']] * batch_meta['batch_size']  # from batch_meta

        task_difficulty = self.maze_difficulty(solutions, self.dataset_meta['difficulty_descriptors'])

        seed = [0] * batch_meta['batch_size']  # TODO: implement

        # task structure one-hot vector [0,1]
        task_structure = np.zeros((batch_meta['batch_size'], len(self.dataset_meta['task_structure_descriptors'])),
                                  dtype=int)
        batch_task_structure_idx = self.dataset_meta['task_structure_descriptors'].index(batch_meta['task_structure'])
        task_structure[:, batch_task_structure_idx] = 1

        # TODO: make robust against change of label descriptors (or declare label descriptors as a class var)
        # TODO: assert checks that they are all the right dimensions
        label_ids = np.arange(batch_meta['batch_size'])
        label_contents = {0: task_difficulty, 1: task_structure, 2: batch_id, 3: seed, 4: solutions}

        return label_ids, label_contents

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

    def maze_difficulty(self, solutions: List[List[Tuple]], difficulty_descriptors: List[str]) -> np.ndarray:

        difficulty_metrics = np.zeros((len(solutions), len(difficulty_descriptors)))

        if 'shortest_path' in difficulty_descriptors:
            shortest_path_ind = difficulty_descriptors.index('shortest_path')
            shortest_path_length = [len(solutions[i]) for i in range(len(solutions))]
            difficulty_metrics[:, shortest_path_ind] = shortest_path_length

        # TODO: add other difficulty metrics

        return difficulty_metrics

    def normalise_difficulty(self):
        pass  # TODO: implement

    def encode_mazes(self, mazes: Union[Maze, List[Maze]]) -> np.ndarray:

        if isinstance(mazes, Maze):
            mazes = [mazes]

        # Obtain the different channels
        grids = np.array([mazes[i].grid for i in range(len(mazes))])
        start_positions_indices = np.array([[i, mazes[i].start[0], mazes[i].start[1]] for i in range(len(mazes))])
        goal_positions_indices = np.array([[i, mazes[i].end[0], mazes[i].end[1]] for i in range(len(mazes))])
        start_position_channels, goal_position_channels = (np.zeros(grids.shape) for i in range(2))
        start_position_channels[tuple(start_positions_indices.T)] = 1
        goal_position_channels[tuple(goal_positions_indices.T)] = 1

        # flatten
        # grids = grids.reshape(grids.shape[0], -1)
        # start_position_channels = start_position_channels.reshape(start_position_channels.shape[0], -1)
        # goal_position_channels = goal_position_channels.reshape(goal_position_channels.shape[0], -1)

        # merge
        features = np.stack((grids, start_position_channels, goal_position_channels), axis=-1)

        return features

        # np.array([[i, starts[i][0], starts[i][1]] for i in range(len(starts))])

    def decode_mazes(self, mazes: np.ndarray, config: Dict[str, Any]) -> List[Maze]:
        raise NotImplementedError

    def decode_minigrid(self, mazes: np.ndarray, config: Dict) -> List[Any]:
        raise NotImplementedError


if __name__ == '__main__':
    dataset_meta = {
        'output_file': 'dataset.meta',
        'maze_size': (27, 27),  # TODO: assert odd
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
            'rooms_unstructured',
            'rooms_structured',
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
            'batch_size': 10000,
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

    dataset_directory = 'only_grid_' + str(batches_meta[0]['batch_size']) + 'x' + str(dataset_meta['maze_size'][0])
    MazeGenerator = MazeDatasetGenerator(dataset_meta=dataset_meta, batches_meta=batches_meta, save_dir=dataset_directory)
    MazeGenerator.generate_data()
    print("Done")
