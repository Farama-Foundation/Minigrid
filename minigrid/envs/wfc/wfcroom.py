import logging

from omegaconf import DictConfig
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import pickle
from pathlib import Path
import os
import dgl

from mazelib import Maze
from mazelib.generate.Prims import Prims
from gym_minigrid.envs.multiroom_mod import MultiRoomEnv

from util import graph_metrics
import util.transforms as tr
import util.util as util

logger = logging.getLogger(__name__)

# Map of object type to channel and id used within that channel, used for grid and gridworld representations
# Agent and Start are considered equivalent
OBJECT_TO_CHANNEL_AND_IDX = {
    'empty'         : (0, 0),
    'wall'          : (0, 1),
    'agent': (1, 1),
    'start': (1, 1),
    'goal'          : (2, 1),
}

# Map of object type to feature dimension, used for graph representations
# Agent and Start are considered equivalent
OBJECT_TO_FEATURE_DIM = {
    'empty'         : 0,
    'wall'          : 1,
    'agent'         : 2,
    'start'         : 2,
    'goal'          : 3,
}

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def generate_dataset(cfg: DictConfig) -> None:
    """
    Generate a dataset of gridworld navigation problems.
    """

    dataset_config = cfg.data.data_generation
    logger.info("\n" + OmegaConf.to_yaml(dataset_config))

    logger.info("Parsing Data Generation config to DatasetGenerator...")
    DatasetGenerator = GridNavDatasetGenerator(dataset_config, dataset_dir_name=dataset_config.dir_name)
    logger.info("Generating Data...")
    DatasetGenerator.generate_dataset(normalise_metrics=dataset_config.normalise_metrics)
    logger.info("Done.")


class GridNavDatasetGenerator:

    def __init__(self, dataset_config: DictConfig, dataset_dir_name=None):
        self.config = dataset_config
        self.dataset_meta = self.get_dataset_metadata()
        self.batches_meta = self.get_batches_metadata()
        self.data_type = self.dataset_meta['data_type']
        self.save_dir = self.get_dataset_dir(dataset_dir_name)
        self.generated_batches = []
        self.generated_labels = []
        self.generated_label_contents = []

        if self.config.label_descriptors_config.use_seed:
            if self.config.seed is None:
                self.seed = self.config.label_descriptors_config.seed = 123456 #default seed
            else:
                self.seed = self.config.seed
            util.seed_everything(self.seed)
            logger.info(f"Using seed {self.seed}")

        self._task_seeds = torch.randperm(int(1e7)) #10M possible seeds


    def generate_dataset(self, normalise_metrics: bool = True):

        n_generated_samples = 0
        for i, batch_meta in enumerate(self.batches_meta):
            logger.info(f"Generating Batch {i}/{len(self.batches_meta)}")
            batch_g = BatchGenerator(batch_meta, self.dataset_meta, seeds=
                self._task_seeds[n_generated_samples:n_generated_samples+batch_meta['batch_size']])
            batch_features, batch_label_ids, batch_label_contents = batch_g.generate_batch()
            batch_label_ids += n_generated_samples
            n_generated_samples += len(batch_label_ids)
            self.generated_batches.append(batch_features)
            self.generated_labels.append(batch_label_ids)
            self.generated_label_contents.append(batch_label_contents)

        self.config.size = n_generated_samples

        if normalise_metrics: self.normalise_metrics()

        #update the dataset meta with the values automatically updated during generation
        self.dataset_meta = self.get_dataset_metadata()

        # creates folder if it does not exist.
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for i in range(len(self.generated_batches)):
            self.save_batch(self.generated_batches[i], self.generated_labels[i],
                            self.generated_label_contents[i], self.batches_meta[i])

        self.save_dataset_meta()

    def normalise_metrics(self):

            maximums = ["shortest_path", "resistance", "navigable_nodes"]
            for key in maximums:
                stack = []
                for i in range(len(self.generated_label_contents)):
                    stack.append(self.generated_label_contents[i][key])
                stack = torch.cat(stack)
                self.config.label_descriptors_config[key]['normalisation_factor'] = \
                    float(stack.amax() * self.config.label_descriptors_config[key]['max'])
                for i in range(len(self.generated_label_contents)):
                    self.generated_label_contents[i][key] = \
                        self.generated_label_contents[i][key].to(torch.float) \
                        / self.config.label_descriptors_config[key]['normalisation_factor']

    def get_dataset_metadata(self):
        dataset_meta = {
            'output_file'                      : 'dataset.meta',
            'seed'                             : self.config.seed,
            'data_type'                        : self.config.data_type,  # types: gridworld, grid, graph
            'encoding'                         : self.config.encoding,  # types: minimal, full (only for graph)
            'data_dim'                         : (self.config.gridworld_data_dim[-1],)*2,  # TODO: assert odd. Note: always in "gridworld" type
            'task_type'                        : self.config.task_type,
            'label_descriptors':                 self.config.label_descriptors,
            'label_descriptors_config'         : self.config.label_descriptors_config,
            'feature_descriptors'              : self.config.feature_descriptors,
            }

        return dataset_meta

    def get_batches_metadata(self):

        all_batches_meta = {
            "train": [],
            "test": []
            }

        for regime, batches_meta in all_batches_meta.items():

            batch_ids = []
            if regime == "train":
                num_batches = self.config.num_train_batches
                batch_size = self.config.size_train_batch
                if self.config.train_batch_ids is None: self.config.train_batch_ids = batch_ids
                start_at_id = 0
            elif regime == "test":
                num_batches = self.config.num_test_batches
                batch_size = self.config.size_test_batch
                if self.config.test_batch_ids is None: self.config.test_batch_ids = batch_ids
                start_at_id = 90
            else:
                raise NotImplementedError(f"Regime {regime} not implemented.")

            if num_batches > 0:

                task_structures = []
                assigned_batches = 0
                for task, split in self.config.task_structure_split.items():
                    assert num_batches % split == 0, \
                        f"Cannot split provided num_{regime}_batches ({num_batches}) by split ratio " \
                        f"{split} for task {task}. Adjust data_generation config."
                    task_num_batches = int(split*num_batches)
                    task_structures.extend([task]*task_num_batches)
                    assigned_batches += task_num_batches

                assert assigned_batches == num_batches, \
                    f"Provided task structure split {str(self.config.task_structure_split)} is not compatible with " \
                    f"num_{regime}_batches ({num_batches}). Adjust data_generation config."
                for i in range(num_batches):
                    batch_id = i + start_at_id
                    batch_ids.append(batch_id)

                    if task_structures[i] == "maze":
                        algo = "Prims"
                    elif task_structures[i] == "rooms_unstructured_layout":
                        algo = "Minigrid_Multiroom"
                    else:
                        raise NotImplementedError(f'No task generation algorithm available for task structure '
                                                  f'"{task_structures[i]}" ')

                    if batch_id >= 90:
                        prefix = "test_"
                    else:
                        prefix = ""

                    batch_meta = {
                        'output_file': f'{prefix}batch_{batch_id}.data',
                        'batch_size': batch_size,
                        'batch_id': batch_id,
                        'task_structure': task_structures[i],
                        'generating_algorithm': algo,
                        'generating_algorithm_options': [],
                        }
                    batches_meta.append(batch_meta)

        batches_meta = []
        for val in all_batches_meta.values():
            batches_meta.extend(val)


        return batches_meta

    def get_dataset_dir(self, dir_name=None):
        base_dir = str(Path(__file__).resolve().parent) + '/datasets/'

        if dir_name is None:
            task_structures = '-'.join(sorted(self.config.label_descriptors_config.task_structures))
            dir_name = f"ts={task_structures}-x={self.dataset_meta['data_type']}-s={self.config.size}" \
                                f"-d={self.config.gridworld_data_dim[-1]}-f={len(self.config.feature_descriptors)}" \
                                f"-enc={self.config.encoding}"
        return base_dir + dir_name + '/'

    def save_batch(self, batch_data: List[Any], batch_labels: np.ndarray,
                   batch_label_contents: Dict[int, Any], batch_meta: Dict[str, Any]):

        filename = self.save_dir + batch_meta['output_file']
        if isinstance(batch_labels, np.ndarray):
            batch_labels = torch.tensor(batch_labels)

        # need to save (data, labels) and (label_contents, metadata) in 2 separate files because of limitations of
        # save_graphs()
        logger.info(f"Saving Batch {batch_meta['batch_id']} to {filename}.")
        if self.data_type=='graph':
            entry = {'label_contents': batch_label_contents, 'batch_meta': batch_meta}
            dgl.data.utils.save_graphs(filename, batch_data, {'labels': batch_labels})
            filename += '.meta'
            logger.info(f"Saving Batch {batch_meta['batch_id']} Metadata to {filename}.")
        else:
            entry = {'data': batch_data, 'labels': batch_labels,
                     'label_contents': batch_label_contents, 'batch_meta': batch_meta}
        with open(filename, 'wb') as f:
            pickle.dump(entry, f)

    def save_dataset_meta(self):

        entry = self.dataset_meta

        filename = self.save_dir + self.dataset_meta['output_file']
        logger.info(f"Saving dataset metadata to {filename}.")
        with open(filename, 'wb') as f:
            pickle.dump(entry, f)


class BatchGenerator:

    def __new__(cls, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any], seeds:torch.Tensor=None):

        if batch_meta['task_structure'] == 'maze':
            instance = super().__new__(MazeBatch)
        elif batch_meta['task_structure'] == 'rooms_unstructured_layout':
            instance = super().__new__(RoomsUnstructuredBatch)
        else:
            raise KeyError("Task Structure was not recognised")

        instance.__init__(batch_meta, dataset_meta, seeds=seeds)

        return instance

    def __init__(self):
        raise RuntimeError(f"{type(self)} is a Class Factory. Assign it to a variable. ")


class Batch:

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any], seeds:torch.Tensor=None):
        self.seeds = seeds
        self.batch_meta = batch_meta
        self.dataset_meta = dataset_meta
        self.data_type = dataset_meta['data_type']
        self.label_ids = np.arange(self.batch_meta['batch_size'])
        self.label_contents = dict.fromkeys(self.dataset_meta['label_descriptors'])
        self.features = None

    def generate_batch(self):

        features = self.generate_data()
        if not util.check_unique(features).all():
            logger.warning(f"Batch {self.batch_meta['batch_id']} generated duplicate features.")
        if self.data_type == 'gridworld':
            pass
        elif self.data_type == 'grid':
            features = tr.Nav2DTransforms.encode_gridworld_to_grid(features)
            #TODO: cleanup and put as unit test
            # features3 = tr.Nav2DTransforms.encode_grid_to_gridworld(features2)
            # assert np.array_equal(features, features3)
        elif self.data_type == 'graph':
            features = tr.Nav2DTransforms.encode_gridworld_to_graph(features)
            self.generate_label_contents(features)

        self.features = features
        return self.features, self.label_ids, self.label_contents

    def generate_data(self):
        raise NotImplementedError

    def generate_label_contents(self, features):

        for key in self.label_contents.keys():
            self.label_contents[key] = []

        self.label_contents["seed"] = self.seeds if self.seeds is not None else [None] * self.batch_meta['batch_size']
        self.label_contents["task_structure"] = [self.batch_meta["task_structure"]] * self.batch_meta['batch_size']
        self.label_contents["generating_algorithm"] = [self.batch_meta["generating_algorithm"]] * self.batch_meta['batch_size']

        start_dim, goal_dim = self.dataset_meta['feature_descriptors'].index('start'), \
                              self.dataset_meta['feature_descriptors'].index('goal')
        for i, graph in enumerate(features):
            start_node = int(graph.ndata["feat"][:,start_dim].argmax())
            goal_node = int(graph.ndata["feat"][:,goal_dim].argmax())
            graph, solvable = graph_metrics.prepare_graph(graph, start_node, goal_node)

            shortest_paths = graph_metrics.shortest_paths(graph, start_node, goal_node, num_paths=1)
            self.label_contents["optimal_trajectories"].append(shortest_paths)
            self.label_contents["shortest_path"].append(len(shortest_paths[0]))

            resistance_distance = graph_metrics.resistance_distance(graph, start_node, goal_node)
            self.label_contents["resistance"].append(resistance_distance)

            num_navigable_nodes = graph.number_of_nodes()
            self.label_contents["navigable_nodes"].append(num_navigable_nodes)

        self.label_contents["shortest_path"] = torch.tensor(self.label_contents["shortest_path"])
        self.label_contents["resistance"] = torch.tensor(self.label_contents["resistance"])
        self.label_contents["navigable_nodes"] = torch.tensor(self.label_contents["navigable_nodes"])

    #Note: Returns the gridworld in one given permutation


class MazeBatch(Batch):

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any], seeds:torch.Tensor=None):
        super().__init__(batch_meta, dataset_meta, seeds)

    def generate_data(self):
        # Set up maze generator
        maze_generator = Maze()
        maze_size_arg = [int((x - 1) / 2) for x in self.dataset_meta['data_dim']]

        # Set up generating algorithm
        if self.batch_meta['generating_algorithm'] == 'Prims':
            maze_generator.generator = Prims(*maze_size_arg)
        else:
            raise KeyError(f"Maze generating algorithm '{self.batch_meta['generating_algorithm']}' was not recognised")

        batch_features = []
        for i in range(self.batch_meta['batch_size']):
            maze_generator.set_seed(int(self.seeds[i]))
            maze_generator.generate()
            maze_generator.generate_entrances(False, False)
            features = tr.Nav2DTransforms.encode_maze_to_gridworld(maze_generator)
            batch_features.append(features)

        batch_features = np.squeeze(batch_features)
        return batch_features


class RoomsUnstructuredBatch(Batch):

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any], seeds:torch.Tensor=None):
        super().__init__(batch_meta, dataset_meta, seeds)

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, Any]]:
        # Set up generator
        envs = [MultiRoomEnv(minNumRooms=4, maxNumRooms=12, minRoomSize=5, maxRoomSize=9,
                             grid_size=self.dataset_meta['data_dim'][0], odd=True,
                             seed=int(self.seeds[i])) for i in range(self.batch_meta['batch_size'])]

        batch_features = tr.Nav2DTransforms.encode_minigrid_to_gridworld(envs)

        return batch_features

if __name__ == '__main__':
    generate_dataset()