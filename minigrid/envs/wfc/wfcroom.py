import logging

from omegaconf import DictConfig
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import einops
from torchvision import transforms
import pickle
from pathlib import Path
import os
import dgl
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

from util import graph_metrics
from util.util import seed_everything

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
            seed_everything(self.seed)
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

                    batch_meta = {
                        'output_file': f'batch_{batch_id}.data',
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
            task_structures = '-'.join(self.config.label_descriptors_config.task_structures)
            dir_name = f"ts={task_structures}-x={self.dataset_meta['data_type']}-s={self.config.size}" \
                                f"-d={self.config.gridworld_data_dim[-1]}-f={self.config.gridworld_data_dim[0]}" \
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
        if self.data_type == 'gridworld':
            pass
        elif self.data_type == 'grid':
            features = self.encode_gridworld_to_grid(features)
            #TODO: cleanup and put as unit test
            # features3 = self.encode_grid_to_gridworld(features2)
            # assert np.array_equal(features, features3)
        elif self.data_type == 'graph':
            features = self.encode_gridworld_to_graph(features)
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
            graph = graph.to_networkx()

            shortest_paths = graph_metrics.shortest_paths(graph, start_node, goal_node, num_paths=1)
            self.label_contents["optimal_trajectories"].append(shortest_paths)
            self.label_contents["shortest_path"].append(len(shortest_paths[0]))

            resistance_distance = graph_metrics.resistance_distance(graph, start_node, goal_node)
            self.label_contents["resistance"].append(resistance_distance)

            num_navigable_nodes = graph_metrics.len_connected_component(graph, start_node, goal_node)
            self.label_contents["navigable_nodes"].append(num_navigable_nodes)

        self.label_contents["shortest_path"] = torch.tensor(self.label_contents["shortest_path"])
        self.label_contents["resistance"] = torch.tensor(self.label_contents["resistance"])
        self.label_contents["navigable_nodes"] = torch.tensor(self.label_contents["navigable_nodes"])

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
        mazes = [Maze() for i in range(grids.shape[0])]
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
        assert gridworlds.shape[1] % 2 == 1 and gridworlds.shape[2] % 2 == 1, \
            "Inputted Gridworlds do not have a layout of odd dimensions"
        assert gridworlds.shape[-1] == 3, "Inputted Gridworlds do not have 3 channels"
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

        start_channels, goal_channels = (np.zeros(grid_layout_dim[:-1]) for i in range(2))
        start_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['start'][0]]
                                        == OBJECT_TO_CHANNEL_AND_IDX['start'][1])
        start_inds_grid = (start_inds_gridworld[0], np.floor(start_inds_gridworld[1] / 2).astype(int),
                           np.floor(start_inds_gridworld[2] / 2).astype(int))
        goal_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['goal'][0]]
                                       == OBJECT_TO_CHANNEL_AND_IDX['goal'][1])
        goal_inds_grid = (goal_inds_gridworld[0], np.floor(goal_inds_gridworld[1] / 2).astype(int),
                           np.floor(goal_inds_gridworld[2] / 2).astype(int))

        start_channels[start_inds_grid] = 1
        goal_channels[goal_inds_grid] = 1

        # merge
        grids = np.stack((grid_layouts[...,0], grid_layouts[...,1], start_channels, goal_channels), axis=-1)
        return grids

    @staticmethod
    def encode_grid_to_gridworld(grids: Union[np.ndarray, torch.Tensor], layout_only=False):

        if layout_only: expected_channels = 2
        else: expected_channels = 4

        #TODO: perf remodel to handle GPU
        tensor = False
        if torch.is_tensor(grids):
            tensor = True
            device = grids.device
            assert len(grids.shape) == 4, f"Grids Tensor has {len(grids.shape)} dimensions. Expected {4}"
            if grids.shape[-1] != expected_channels and grids.shape[1] == expected_channels:
                grids = torch.permute(grids, (0, 2, 3, 1)) # (B, C, H, W) -> (B, H, W, C)
            grids = grids.detach().cpu().numpy()

        assert grids.shape[-1] == expected_channels, f"Inputted Grids have {grids.shape[-1]} channels. Expected {expected_channels}"

        gridworlds_layout_dim = (
        grids.shape[0], int(2 * grids.shape[1] + 1), int(2 * grids.shape[2] + 1))
        gridworlds_layouts = np.ones(gridworlds_layout_dim) * OBJECT_TO_CHANNEL_AND_IDX['wall'][-1]

        gridworlds_layout_channel = OBJECT_TO_CHANNEL_AND_IDX['empty'][0] #could be used with a "sort"
        gridworlds_empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]

        for m in range(grids.shape[0]):
            for i in range(grids.shape[1]):
                for j in range(grids.shape[2]):
                    if grids[m,i,j,0] == 1 or grids[m,i,j,1] == 1:
                        i_gridworld, j_gridworld = (2 * i + 1, 2 * j + 1)
                        gridworlds_layouts[m,i_gridworld,j_gridworld] = gridworlds_empty_idx
                        if grids[m,i,j,0] == 1:
                            gridworlds_layouts[m, i_gridworld, j_gridworld + 1] = gridworlds_empty_idx
                            gridworlds_layouts[m, i_gridworld, j_gridworld + 2] = gridworlds_empty_idx
                        if grids[m,i,j,1] == 1:
                            gridworlds_layouts[m, i_gridworld + 1, j_gridworld] = gridworlds_empty_idx
                            gridworlds_layouts[m, i_gridworld + 2, j_gridworld] = gridworlds_empty_idx
                        # clique rule
                        if grids[m,i,j,0] == grids[m,i,j,1] == grids[m,i+1,j,0] == grids[m,i,j+1,1] == 1:
                            gridworlds_layouts[m, i_gridworld + 1, j_gridworld + 1] = gridworlds_empty_idx

        if layout_only:
            gridworlds = np.reshape(gridworlds_layouts, (*gridworlds_layouts.shape,1))
        #DROPED: use object dictionary
        else:
            start_channels, goal_channels = (np.zeros(gridworlds_layout_dim) for i in range(2))

            start_inds_grids = np.where(grids[..., 2] == 1)
            start_inds_gridworlds = (start_inds_grids[0], (2*start_inds_grids[1] + 1).astype(int),
                               (2*start_inds_grids[2] + 1).astype(int))
            goal_inds_grids = np.where(grids[..., 3] == 1)
            goal_inds_gridworlds = (goal_inds_grids[0], (2*goal_inds_grids[1] + 1).astype(int),
                               (2*goal_inds_grids[2] + 1).astype(int))

            start_channels[start_inds_gridworlds] = OBJECT_TO_CHANNEL_AND_IDX['start'][1]
            goal_channels[goal_inds_gridworlds] = OBJECT_TO_CHANNEL_AND_IDX['goal'][1]

            # merge
            gridworlds = np.stack((gridworlds_layouts, start_channels, goal_channels), axis=-1)

        if tensor:
            gridworlds = torch.tensor(gridworlds, dtype=torch.float, device=device)
            gridworlds = torch.permute(gridworlds, (0, 3, 1, 2)) # (B, H, W, C) -> (B, C, H, W)

        return gridworlds

    @staticmethod
    def encode_gridworld_to_graph(gridworlds: np.ndarray):
        # Graph feature shape [empty, wall, start, goal]
        # Graph nodes: ((gw_dim - 1)/2)**2

        # gridworlds shape: [m, odd, odd, 3]
        assert gridworlds.shape[1] % 2 == 1 and gridworlds.shape[2] % 2 == 1, \
            "Inputted Gridworlds do not have a layout of odd dimensions"
        assert gridworlds.shape[-1] == 3, "Inputted Gridworlds do not have 3 channels"

        dim_grid = (int((gridworlds.shape[1] - 1) / 2), int((gridworlds.shape[2] - 1) / 2))

        layout_channel = OBJECT_TO_CHANNEL_AND_IDX['empty'][0]
        empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]

        adj = Batch.encode_gridworld_layout_to_adj(gridworlds[...,layout_channel], empty_idx) # M, N, N

        start_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['start'][0]]
                                        == OBJECT_TO_CHANNEL_AND_IDX['start'][1])
        goal_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['goal'][0]]
                                       == OBJECT_TO_CHANNEL_AND_IDX['goal'][1])

        start_nodes_graph = (start_inds_gridworld[0],
                             ((start_inds_gridworld[1] - 1)/2 * dim_grid[0] + (start_inds_gridworld[2] - 1)/2).astype(int))
        goal_nodes_graph = (goal_inds_gridworld[0],
                             ((goal_inds_gridworld[1] - 1)/2 * dim_grid[0] + (goal_inds_gridworld[2] - 1)/2).astype(int))
        active_nodes_graph = np.where(adj.sum(axis=1)!=0)
        wall_nodes_graph = np.where(adj.sum(axis=1)==0)


        feats = np.zeros((adj.shape[0], adj.shape[1], 4)) # M, N, D
        feats[(*start_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['start']]*len(start_nodes_graph[0])))] = 1
        feats[(*goal_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['goal']]*len(goal_nodes_graph[0])))] = 1
        if wall_nodes_graph[0].size != 0: #only if array not empty.
            feats[(*wall_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['wall']]*len(wall_nodes_graph[0])))] = 1
        # empty features are the active nodes, removing the nodes having goal or start feature
        feats[(*active_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['empty']]*len(active_nodes_graph[0])))] = 1
        feats[(*goal_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['empty']] * len(goal_nodes_graph[0])))] = 0
        feats[(*start_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['empty']] * len(start_nodes_graph[0])))] = 0
        feats = torch.tensor(feats)
        # check all features are one-hot.
        assert (feats.sum(axis=-1) == 1).all()

        graphs = []
        for m in range(adj.shape[0]):
            src, dst = np.nonzero(adj[m])
            g = dgl.graph((src, dst), num_nodes=len(feats[m]))
            g.ndata['feat'] = feats[m]
            graphs.append(g)

        return graphs

    #Note: Returns the gridworld in one given permutation
    @staticmethod
    def encode_graph_to_gridworld(graphs: Union[dgl.DGLGraph, List[dgl.DGLGraph], tuple],
                                  attributes:Tuple[str]=("empty", "wall", "start", "goal"),
                                  used_attributes:Tuple[str]=("start", "goal" ),
                                  probabilistic_mode: bool = False,
                                  output_dtype: str = 'tensor'):
    #TODO: probabilistic_mode not fully implemented
    #TODO: perf enhancements if going full tensor

        def get_gw_inds(nodes_tuple:Tuple[np.ndarray], n_nodes, mapping=lambda x : 2*x+1):
            inds_tuple = []
            for nodes in nodes_tuple:
                inds = (nodes[1] // np.sqrt(n_nodes), nodes[1] % np.sqrt(n_nodes))
                inds = tuple([mapping(i.astype(int)) for i in inds])
                inds = (nodes[0],) + inds
                inds_tuple.append(inds)
            return tuple(inds_tuple)

        #Note: modes 2 and 3 can only work for layouts with 1-to-1 cell-node characterisation
        possible_modes = {
            ():0, ("",):0,                              #0: Layout only from A
            ("start", "goal"): 1,                       #1: Layout from A, start and goal from Fx
            ("empty", "start", "goal"): 2,              #2: Layout, start, goal from Fx
            ("empty", "wall", "start", "goal"): 3,      #3: Layout, start, goal from Fx, may form impossible layouts
        }

        try:
            mode = possible_modes[tuple(used_attributes)]
        except KeyError:
            raise AttributeError(f"Gridworld encoding from {used_attributes} is not possible.")

        if isinstance(graphs, tuple):
            A, Fx = graphs
            #n_nodes = Fx.shape[-2]
            n_nodes = int(A.shape[-1] / 2 + 1) #TODO: find a better way to handle full graph encoding (may require an additional input argument)
            A = torch.reshape(A, (A.shape[0], -1, 2))
            A = A.cpu().numpy() #TODO make more efficient to handle tensors
            if Fx is not None:
                Fx = Fx.cpu().numpy()
        elif isinstance(graphs, dgl.DGLGraph) or isinstance(graphs[0], dgl.DGLGraph):
            if isinstance(graphs, dgl.DGLGraph): graphs = dgl.unbatch(graphs)
            n_nodes = graphs[0].num_nodes() #assumption that all graphs have same number of nodes
            feat_dim = graphs[0].ndata['feat'].shape
            assert n_nodes % np.sqrt(n_nodes) == 0 # we are assuming square layout

            A = np.empty((len(graphs), n_nodes, n_nodes))
            Fx = np.empty((len(graphs), *feat_dim))
            for m in range(len(graphs)):
                A[m] = graphs[m].adj().cpu().to_dense().numpy()
                Fx[m] = graphs[m].ndata['feat'].cpu().numpy()
            A = Batch.encode_adj_to_reduced_adj(A)
        else:
            raise RuntimeError(f"data format {type(graphs)} is not supported by function. Format supported are"
                                 f"List[dgl.DGLGraph], tuple[tensor, tensor]")

        if output_dtype == 'tensor':
            device = graphs[0].device

        gridworld_layout_dim = (int(2 * np.sqrt(n_nodes) + 1), int(2 * np.sqrt(n_nodes) + 1))

        # Modes for which we need A
        if mode in [0, 1]:
            gridworlds_layouts = Batch.encode_reduced_adj_to_gridworld_layout(A, gridworld_layout_dim, probalistic_mode=probabilistic_mode)
            if mode in [0, ]:
                gridworlds = np.reshape(gridworlds_layouts, (*gridworlds_layouts.shape, 1))
        # Modes for which we need Fx[start, goal]
        if mode in [1, 2,]: #[1,2,3] when implemented
            gridworlds = np.zeros((Fx.shape[0], *gridworld_layout_dim, 3))

            start_nodes = np.where(Fx[..., attributes.index('start')] == 1)
            goal_nodes = np.where(Fx[..., attributes.index('goal')] == 1)
            # start_inds = (start_nodes[1] // np.sqrt(n_nodes), start_nodes[1] % np.sqrt(n_nodes))
            # goal_inds = (goal_nodes[1] // np.sqrt(n_nodes), goal_nodes[1] % np.sqrt(n_nodes))
            # start_inds, goal_inds = (tuple([2 * i.astype(int) + 1 for i in tup]) for tup in (start_inds, goal_inds))
            # start_inds = (start_nodes[0],) + start_inds
            # goal_inds = (goal_nodes[0],) + goal_inds

            start_inds, goal_inds = get_gw_inds((start_nodes, goal_nodes), n_nodes)

            gridworlds[(*start_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['start'][0]] * gridworlds.shape[0]))] = \
                OBJECT_TO_CHANNEL_AND_IDX['start'][1]
            gridworlds[(*goal_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['goal'][0]] * gridworlds.shape[0]))] = \
                OBJECT_TO_CHANNEL_AND_IDX['goal'][1]
            if mode in [1,]: #add layout from adjacency
                gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['wall'][0]] = OBJECT_TO_CHANNEL_AND_IDX['wall'][1] * \
                                                                    gridworlds_layouts
            elif mode in [2,]: #add layout from empty nodes
                # set all cells to wall in layout channel
                gridworlds[..., np.array([OBJECT_TO_CHANNEL_AND_IDX['wall'][0]])] = OBJECT_TO_CHANNEL_AND_IDX['wall'][1]

                # set all non wall (empty, goal, start) to empty
                #start, goal
                gridworlds[(*start_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * gridworlds.shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]
                gridworlds[(*goal_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * gridworlds.shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]

                #empty
                empty_nodes = np.where(Fx[..., attributes.index('empty')] == 1)
                # empty_inds = (empty_nodes[1] // np.sqrt(n_nodes), empty_nodes[1] % np.sqrt(n_nodes))
                # empty_inds = tuple([2 * i.astype(int) + 1 for i in empty_inds])
                # empty_inds = (empty_nodes[0],) + goal_inds
                empty_inds, = get_gw_inds((empty_nodes,), n_nodes)

                gridworlds[(*empty_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * empty_inds[0].shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]
        elif mode in [3,]:
            raise NotImplementedError(f"Gridworld encoding from {used_attributes} not yet implemented.")

        if output_dtype == 'tensor':
            gridworlds = torch.tensor(gridworlds, dtype=torch.float, device=device)
            gridworlds = torch.permute(gridworlds, (0, 3, 1, 2)) # (B, H, W, C) -> (B, C, H, W)

        return gridworlds

    @staticmethod
    def encode_reduced_adj_to_gridworld_layout(A: Union[np.ndarray, torch.tensor], layout_dim, probalistic_mode=False, prob_threshold=0.5):

        n_nodes = A.shape[1] + 1
        gridworlds_layout_dim = (A.shape[0], *layout_dim)
        assert gridworlds_layout_dim == (A.shape[0], int(2 * np.sqrt(n_nodes) + 1), int(2 * np.sqrt(n_nodes) + 1))
        gridworlds_layouts = np.zeros(gridworlds_layout_dim)

        for m in range(A.shape[0]):
            for n in range(A[m].shape[0]):
                i_n = n // int(np.sqrt(n_nodes))
                j_n = n % int(np.sqrt(n_nodes))
                i, j = 2 * i_n + 1, 2 * j_n + 1
                # nodes:
                cell_val = max(A[m, n, :].max(), gridworlds_layouts[m, i, j]) # node likelihood is max of (max edge probability and previous inputed max edge probability)
                gridworlds_layouts[m, i, j] = cell_val
                # horizontal edges
                if (j+2)<gridworlds_layouts.shape[2]:
                    edge_val = A[m, n, 0]
                    cell_val = max(A[m, n, 0], gridworlds_layouts[m, i, j + 2]) #update node likelihood if an edge of higher probability is found
                    gridworlds_layouts[m, i, j + 1] = edge_val
                    gridworlds_layouts[m, i, j + 2] = cell_val
                # vertical edges
                if (i+2)<gridworlds_layouts.shape[1]:
                    edge_val = A[m, n, 1]
                    cell_val = max(A[m, n, 1], gridworlds_layouts[m, i + 2, j])
                    gridworlds_layouts[m, i + 1, j] = edge_val
                    gridworlds_layouts[m, i + 2, j] = cell_val
                # clique rule
                if i + 1 < gridworlds_layouts.shape[1] and j + 1 < gridworlds_layouts.shape[2]:
                    if n+int(np.sqrt(n_nodes)) < A[m].shape[0]:
                        clique = torch.tensor([A[m,n,0],A[m,n,1],A[m,n+1,1],A[m,n+int(np.sqrt(n_nodes)),0]])
                        cell_val = clique.min()
                        gridworlds_layouts[m, i + 1, j + 1] = cell_val

        if torch.is_tensor(A):
            gridworlds_layouts = torch.tensor(gridworlds_layouts).to(A)

        if not probalistic_mode:
            wall_idx = OBJECT_TO_CHANNEL_AND_IDX['wall'][-1]
            empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]
            deterministic_layout = (gridworlds_layouts >= prob_threshold) * 1
            gridworlds_layouts[deterministic_layout == 1] = empty_idx
            gridworlds_layouts[deterministic_layout == 0] = wall_idx

        return gridworlds_layouts

    @staticmethod
    def encode_gridworld_layout_to_adj(layouts: np.ndarray, empty_idx=0):
        #layouts shape: [m, odd, odd]
        assert layouts.shape[1] % 2 == 1 and layouts.shape[2] % 2 == 1, \
            "Inputted Gridworlds Layouts do not have odd number of elements"
        assert len(layouts.shape) == 3, "Layout not inputted correctly. Input layouts as (m, row, col)"
        node_inds_i, node_inds_j = [i for i in range(1, layouts.shape[1], 2)], [i for i in range(1, layouts.shape[2], 2)]
        A = np.zeros((layouts.shape[0], len(node_inds_i) * len(node_inds_j), len(node_inds_i) * len(node_inds_j)))

        for m in range(A.shape[0]):
            for i_A, i_gw in enumerate(node_inds_i):
                for j_A, j_gw in enumerate(node_inds_j):
                    if layouts[m, i_gw, j_gw] == empty_idx:
                        ind_gw_right = (m, i_gw, j_gw + 1)
                        ind_gw_bot = (m, i_gw + 1, j_gw)
                        if layouts[ind_gw_right] == empty_idx:
                            ind_A_right = (m, i_A*len(node_inds_i)+j_A, i_A*len(node_inds_i)+j_A+1)
                            A[ind_A_right] = 1
                        if layouts[ind_gw_bot] == empty_idx:
                            ind_A_bot = (m, i_A*len(node_inds_i)+j_A, (i_A+1)*len(node_inds_i)+j_A)
                            A[ind_A_bot] = 1
            A[m] = np.triu(A[m]) + np.tril(A[m].T, 1)

        return A

    @staticmethod
    def encode_adj_to_reduced_adj(adj: Union[np.ndarray, torch.tensor]):
        # the last sqrt(n)-1 col edges will always be 0.
        #adj shape (m, n, n)
        if torch.is_tensor(adj):
            A = torch.zeros((adj.shape[0], adj.shape[1] - 1, 2)).to(adj.device)
        elif isinstance(adj, np.ndarray):
            A = np.zeros((adj.shape[0], adj.shape[1] - 1, 2))
        dim_grid = int(np.sqrt(adj.shape[1])) #only for square grids
        A[...,0] = adj.diagonal(1,1,2) # row edges
        A[:,:-dim_grid+1,1] = adj.diagonal(dim_grid, 1, 2) # col edges

        return A # (m, n-1, 2)

    @staticmethod
    def encode_reduced_adj_to_adj(adj_r: np.ndarray):
        # only for square grids
        #adj shape (m, n-1, 2)
        A = np.empty((adj_r.shape[0], adj_r.shape[1] + 1, adj_r.shape[1] + 1))
        dim_grid = int(np.sqrt(A.shape[1]))
        for m in range(A.shape[0]):
            A[m] = np.diag(adj_r[m,:,0], k = 1)
            A[m] += np.diag(adj_r[m,:-dim_grid+1,1], k = dim_grid)
            A[m] = np.triu(A[m]) + np.tril(A[m].T, 1)

        return A # (m, n, n)

    @staticmethod
    def augment_adj(n: int, transforms: torch.tensor):
        # transforms represent all allowable permutations in a 2D grid
        nodes_inds = torch.arange(0, n, dtype=torch.int) # node indices in adjacency matrix
        i_n, j_n = nodes_inds.div(int(n**.5), rounding_mode='floor'), nodes_inds % int(n**.5) #corresponding indices in grid space, top left corner origin
        ij_n = torch.stack([i_n, j_n], dim=0) # D N
        ij_n = einops.repeat(ij_n, 'd n ->  p d n', p=transforms.shape[0]) # P D N
        c = torch.tensor([int((n ** .5 - 1) / 2), int((n ** .5 - 1) / 2)], dtype=torch.int).unsqueeze(1) # D 1
        c = einops.repeat(c, 'd n -> p d n', p=transforms.shape[0]) # P D 1
        ij_c = (ij_n - c) # P D=2 N. Corresponding indices with origin in the middle of the grid
        # coordinate transform 1 done: origin in grid space
        ij_t = torch.matmul(transforms, ij_c) # P 2 2 @ P D=2 N -> P D=2 N # rotate the coordinate axis
        # coordinate transform 2 done: axis rotated
        ij_f = (ij_t + c) # add the centroid to come back to a top left corner coordinate system
        ij2n = torch.tensor([int(n**0.5), 1], dtype=torch.int).unsqueeze(1) # D 1 #transformation matrix to get node ordering, left to right, top to bottom in graph space
        ij2n = einops.repeat(ij2n, 'd n -> p n d', p=transforms.shape[0]) # P 1 D
        nodes_inds_t = torch.matmul(ij2n, ij_f).squeeze() # P 1 D @ P D N = P D N #recover the transformed indices
        return nodes_inds_t


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
            features = self.encode_maze_to_gridworld(maze_generator)
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

        batch_features = self.encode_minigrid_to_gridworld(envs)

        return batch_features

if __name__ == '__main__':
    generate_dataset()