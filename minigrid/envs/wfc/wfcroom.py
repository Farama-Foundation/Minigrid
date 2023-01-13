from __future__ import annotations

__package__ = "maze_representations"

import copy
import logging
import time
from bounded_pool_executor import BoundedProcessPoolExecutor
import concurrent.futures
import networkx as nx
from torchvision import utils as vutils
from omegaconf import DictConfig
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import pickle
from pathlib import Path
import os
import dgl
import imageio

from util import make_grid_with_labels
import data_generation.generation_algorithms.wfc_2019f.wfc.wfc_control as wfc_control
import data_generation.generation_algorithms.wfc_2019f.wfc.wfc_solver as wfc_solver
from .util import graph_metrics
from .util import transforms as tr
from .util import util as util
from .util.multiprocessing import MockProcessPoolExecutor, Parser

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def generate_dataset(cfg: DictConfig) -> None:
    """
    Generate a dataset of gridworld navigation problems.
    """

    dataset_config = cfg.data_generation
    logger.info("\n" + OmegaConf.to_yaml(dataset_config))

    logger.info("Parsing Data Generation config to DatasetGenerator...")
    DatasetGenerator = GridNavDatasetGenerator(dataset_config, dataset_dir_name=dataset_config.dir_name,
                                               num_workers=cfg.num_cpus,
                                               multiprocessing=cfg.multiprocessing,
                                               debug_multiprocessing=cfg.debug_multiprocessing)
    logger.info("Generating Data...")
    DatasetGenerator.generate_dataset(normalise_metrics=dataset_config.normalise_metrics)
    logger.info("Done.")


class GridNavDatasetGenerator:

    def __init__(self, dataset_config: DictConfig, dataset_dir_name=None, num_workers=0, multiprocessing=False,
                 debug_multiprocessing=False):
        self.config = self.generate_config(dataset_config)
        self.dataset_meta = self.get_dataset_metadata()
        self.batches_meta = self.get_batches_metadata()
        self.data_type = self.dataset_meta['data_type']
        self.save_dir = self.get_dataset_dir(dataset_dir_name)
        self.generated_batches = {}
        self.use_multiprocessing = True if num_workers != 1 and multiprocessing else False
        self.multi_processing_debug_mode = True if multiprocessing and num_workers != 1 and debug_multiprocessing \
            else False
        self.num_workers = num_workers

        if self.config.label_descriptors_config.use_seed:
            if self.config.seed is None:
                self.seed = self.config.label_descriptors_config.seed = 123456 #default seed
            else:
                self.seed = self.config.seed
            util.seed_everything(self.seed)
            logger.info(f"Using seed {self.seed}")

        self._task_seeds = torch.randperm(int(1e7)) #10M possible seeds
        self._seeds_per_batch = len(self._task_seeds) // len(self.batches_meta)
        self.batch_label_offset = []

    def generate_config(self, dataset_config: DictConfig) -> DictConfig:
        """
        Generate a config for the dataset generator.
        """
        if not dataset_config.generate_images:
            dataset_config.label_descriptors.remove('images')
        if dataset_config.num_train_batches is None:
            dataset_config.num_train_batches = int(np.array(list(dataset_config.task_structure_split.values())).sum())
        if dataset_config.num_test_batches is None:
            dataset_config.num_test_batches = int(np.array(list(dataset_config.task_structure_split.values())).sum())

        return dataset_config

    def generate_dataset(self, normalise_metrics: bool = True):

        n_labels = 0
        seed_counter = 0
        args = []
        for i, batch_meta in enumerate(self.batches_meta):
            self.batch_label_offset.append(n_labels)
            seeds = self._task_seeds[seed_counter:seed_counter+self._seeds_per_batch]
            seed_counter += self._seeds_per_batch
            args.append((batch_meta, self.dataset_meta, seeds, n_labels))
            n_labels += batch_meta['batch_size']

        self.config.size = n_labels

        # creates folder if it does not exist.
        if not os.path.exists(self.save_dir):
            logger.info(f"Saving dataset in new directory {self.save_dir}")
            os.makedirs(self.save_dir)

        if self.use_multiprocessing:
            logger.info(f"Using multiprocessing with {self.num_workers} workers.")
            multi_processing_parser = Parser(GridNavDatasetGenerator._generate_batch_data)
            if not self.multi_processing_debug_mode:
                num_cpus = self.num_workers if self.num_workers != 0 else None #None means use all available cpus
                torch.set_num_threads(1)
                proc = BoundedProcessPoolExecutor(max_workers=num_cpus)
            else: #TODO: fix this
                proc = MockProcessPoolExecutor()
        else:
            proc = None

        # TODO: consider using tqdm to show progress
        if proc is not None:
            with proc as executor:
                dataset = [executor.submit(multi_processing_parser, arg) for arg in args]
                for batch_data in concurrent.futures.as_completed(dataset):
                    try:
                        batch = batch_data.result()
                        self.generated_batches[batch.batch_meta['batch_id']] = batch
                    except Exception as e:
                        logger.error(f"Error during multiprocressing. Error message: {str(e)}")
        else:
            for arg in args:
                batch = GridNavDatasetGenerator._generate_batch_data(*arg)
                self.generated_batches[batch.batch_meta['batch_id']] = batch

        self.check_unique()

        if normalise_metrics: self.normalise_metrics()

        #update the dataset meta with the values automatically updated during generation
        self.dataset_meta = self.get_dataset_metadata()

        for batch in self.generated_batches.values():
            self.save_batch(batch)

        self.save_dataset_meta()
        self.save_layout_thumbnail(n_per_batch=self.config.n_thumbnail_per_batch)

    @staticmethod
    def _generate_batch_data(batch_meta, dataset_meta, seeds, batch_label_offset):
        try:
            time_batch_start = time.perf_counter()
            batch = BatchGenerator(batch_meta, dataset_meta, seeds=seeds)
            batch_features, batch_label_ids, batch_label_contents = batch.generate_batch()
            batch_label_ids += batch_label_offset
            time_batch_end = time.perf_counter()
            batch_generation_time = time_batch_end - time_batch_start
            logger.info(f"Batch {batch_meta['batch_id']}, Task structure: {batch_meta['task_structure']} generated in "
                        f"{batch_generation_time:.2f} seconds. "
                        f"Average time per sample: {batch_generation_time / len(batch_label_ids):.2f} seconds.")
        except Exception as e:
            logger.error(f"Error in child process for batch {batch_meta['batch_id']}. Exception: {str(e)}")
            raise e

        data_size = len(pickle.dumps(batch))
        if data_size > 2 * 10 ** 9:
            raise RuntimeError(f'return data of total size {data_size} bytes can not be sent, too large.')

        return batch

    def normalise_metrics(self):

        maximums = ["shortest_path", "resistance", "navigable_nodes"]
        for key in maximums:
            stack = []
            for batch in self.generated_batches.values():
                stack.append(batch.label_contents[key])
            stack = torch.cat(stack)
            self.config.label_descriptors_config[key]['normalisation_factor'] = \
                float(stack.amax() * self.config.label_descriptors_config[key]['max'])
            for batch_id in self.generated_batches.keys():
                self.generated_batches[batch_id].label_contents[key] = \
                    self.generated_batches[batch_id].label_contents[key].to(torch.float) \
                    / self.config.label_descriptors_config[key]['normalisation_factor']

    def get_dataset_metadata(self):
        # TODO: would be easier to just instantiate all metadata as OmegaConf objects
        dataset_meta = {
            'output_file'                      : 'dataset.meta',
            'seed'                             : self.config.seed,
            'data_type'                        : self.config.data_type,  # types: gridworld, grid, graph
            'encoding'                         : self.config.encoding,  # types: minimal, dense (only for graph)
            'data_dim'                         : (self.config.gridworld_data_dim[-1],)*2,
            'task_type'                        : self.config.task_type,
            'label_descriptors':                 self.config.label_descriptors,
            'label_descriptors_config'         : self.config.label_descriptors_config,
            'graph_feature_descriptors'        : self.config.graph_feature_descriptors,
            'minigrid_feature_descriptors'     : self.config.minigrid_feature_descriptors,
            'ensure_connected'                 : self.config.ensure_connected,
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
                prefix = ''
                num_batches = self.config.num_train_batches
                batch_size = self.config.size_train_batch
                if self.config.train_batch_ids is None: self.config.train_batch_ids = batch_ids
            elif regime == "test":
                prefix = 'test_'
                num_batches = self.config.num_test_batches
                batch_size = self.config.size_test_batch
                if self.config.test_batch_ids is None: self.config.test_batch_ids = batch_ids
            else:
                raise NotImplementedError(f"Regime {regime} not implemented.")

            if num_batches > 0:

                task_structures = []
                assigned_batches = 0
                for task, split in self.config.task_structure_split.items():
                    task_num_batches = int(split)
                    task_structures.extend([task]*task_num_batches)
                    assigned_batches += task_num_batches

                assert assigned_batches == num_batches, \
                    f"Provided task structure split {str(self.config.task_structure_split)} is not compatible with " \
                    f"num_{regime}_batches ({num_batches}). Adjust data_generation config."
                for i in range(num_batches):
                    batch_id = regime + '_' + str(i)
                    batch_ids.append(batch_id)

                    if len(self.config.generating_algorithm) > 1:
                        if task_structures[i] == "maze":
                            algo = "Prims"
                        elif task_structures[i] == "rooms_unstructured_layout":
                            algo = "Minigrid_Multiroom"
                        else:
                            raise NotImplementedError(f'No task generation algorithm available for task structure '
                                                      f'"{task_structures[i]}" ')
                    else:
                        algo = self.config.generating_algorithm[0]

                    batch_meta = {
                        'output_file': f'{prefix}batch_{i}.data',
                        'batch_size': batch_size,
                        'batch_id': batch_id,
                        'task_structure': task_structures[i],
                        'generating_algorithm': algo,
                        'generating_algorithm_options': [],
                        }
                    batches_meta.append(batch_meta)

            else:
                raise ValueError(f"Number of batches must be greater than 0.")

        batches_meta = []
        for val in all_batches_meta.values():
            batches_meta.extend(val)

        return batches_meta

    def get_dataset_dir(self, dir_name=None):
        base_dir = str(Path(__file__).resolve().parent) + '/datasets/'

        if dir_name is None:
            task_structures = '-'.join(sorted(self.config.task_structure_split.keys()))
            dir_name = f"ts={task_structures}-x={self.dataset_meta['data_type']}-s={self.config.size}" \
                                f"-d={self.config.gridworld_data_dim[-1]}-gf={len(self.config.graph_feature_descriptors)}" \
                                f"-enc={self.config.encoding}"

            if len(base_dir + dir_name + '/') > 260:
                task_structures = 'many'
                dir_name = f"ts={task_structures}-x={self.dataset_meta['data_type']}-s={self.config.size}" \
                           f"-d={self.config.gridworld_data_dim[-1]}-gf={len(self.config.graph_feature_descriptors)}" \
                           f"-enc={self.config.encoding}"

                logger.warning(f"Dataset directory name too long. Using {dir_name} instead.")

        path = base_dir + dir_name + '/'

        if len(path) > 260:
            raise ValueError("Dataset dir path too long. Please use a shorter path.")

        if os.path.exists(path):
            raise FileExistsError(f"Dataset directory {path} already exists. Aborting.")

        return path

    def save_batch(self, batch: Batch):

        filename = self.save_dir + batch.batch_meta['output_file']
        if isinstance(batch.label_ids, np.ndarray):
            batch.label_ids = torch.tensor(batch.label_ids)

        # need to save (data, labels) and (label_contents, metadata) in 2 separate files because of limitations of
        # save_graphs()
        logger.info(f"Saving Batch {batch.batch_meta['batch_id']} to {filename}.")
        if self.data_type=='graph':
            entry = {'label_contents': batch.label_contents, 'batch_meta': batch.batch_meta}
            dgl.data.utils.save_graphs(filename, batch.features, {'labels': batch.label_ids})
            filename += '.meta'
            logger.info(f"Saving Batch {batch.batch_meta['batch_id']} Metadata to {filename}.")
        else:
            entry = {'data': batch.features, 'labels': batch.label_ids,
                     'label_contents': batch.label_contents, 'batch_meta': batch.batch_meta}
        with open(filename, 'wb') as f:
            pickle.dump(entry, f)

    def save_dataset_meta(self):

        entry = self.dataset_meta

        filename = self.save_dir + self.dataset_meta['output_file']
        logger.info(f"Saving dataset metadata to {filename}.")
        with open(filename, 'wb') as f:
            pickle.dump(entry, f)

    def check_unique(self):

        all_data = []
        for batch in self.generated_batches.values():
            all_data.extend(batch.features)
        features, _ = util.get_node_features(all_data, device=None) # TODO: add device?
        if not util.check_unique(features).all():
            logger.warning(f"There are duplicate features in the dataset.")
            return False
        else:
            return True

    def save_layout_thumbnail(self, n_per_batch=10):

        images = []
        task_structures = []
        for batch in self.generated_batches.values():
            label_content = batch.label_contents

            #Skip repeated values
            if batch.batch_meta['task_structure'] in task_structures:
                continue

            task_structures.extend(label_content['task_structure'][:n_per_batch])
            if 'images' in label_content:
                images.extend(label_content['images'][:n_per_batch])
            else:
                img = batch.get_images([j for j in range(n_per_batch)])
                images.extend(img)

        filename = 'layouts'
        path = os.path.join(self.save_dir, f'{filename}.png')
        images = make_grid_with_labels(images, task_structures, nrow=n_per_batch, normalize=True, limit=None,
                                       channels_first=True)
        logger.info(f"Saving layout thumbnails to {path}.")
        vutils.save_image(images, path)


class BatchGenerator:

    def __new__(cls, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any], seeds:torch.Tensor=None):

        if batch_meta['generating_algorithm'] == 'wave_function_collapse':
            instance = super().__new__(WaveCollapseBatch)
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
        self.encoding = dataset_meta['encoding']
        self.label_ids = np.arange(self.batch_meta['batch_size'])
        self.label_contents = dict.fromkeys(self.dataset_meta['label_descriptors'])
        self.features = None
        self.images = None

    def generate_batch(self):

        batch_data = self.generate_data()
        if isinstance(batch_data, dgl.DGLGraph) or isinstance(batch_data[0], dgl.DGLGraph):
            features, _ = util.get_node_features(batch_data, device=None) # TODO: add device?
        else:
            features = batch_data

        if not util.check_unique(features).all():
            logger.warning(f"Batch {self.batch_meta['batch_id']} generated duplicate features.")
        if self.data_type == 'gridworld':
            pass
        elif self.data_type == 'grid':
            batch_data = tr.Nav2DTransforms.encode_gridworld_to_grid(batch_data)
        elif self.data_type == 'graph':
            if self.encoding == 'minimal':
                if not (isinstance(batch_data, dgl.DGLGraph) or isinstance(batch_data[0], dgl.DGLGraph)):
                    batch_data = tr.Nav2DTransforms.encode_gridworld_to_graph(batch_data)
            elif self.encoding == 'dense':
                if not (isinstance(batch_data, dgl.DGLGraph) or isinstance(batch_data[0], dgl.DGLGraph)):
                    raise TypeError("Output of generate_data() must be a DGLGraph or list of DGLGraphs. For dense "
                                    "encoding")
            else:
                raise NotImplementedError(f"Encoding {self.encoding} not implemented.")
            self.generate_label_contents(batch_data, features=features)

        self.features = batch_data
        return self.features, self.label_ids, self.label_contents

    def generate_data(self):
        raise NotImplementedError

    def generate_label_contents(self, data, features=None):

        if features is None:
            if isinstance(data, dgl.DGLGraph) or isinstance(data[0], dgl.DGLGraph):
                features, _ = util.get_node_features(data, device=None)  # TODO: add device?
            else:
                features = data

        for key in self.label_contents.keys():
            self.label_contents[key] = []

        self.label_contents["seed"] = self.seeds if self.seeds is not None else [None] * self.batch_meta['batch_size']
        self.label_contents["task_structure"] = [self.batch_meta["task_structure"]] * self.batch_meta['batch_size']
        self.label_contents["generating_algorithm"] = [self.batch_meta["generating_algorithm"]] * self.batch_meta['batch_size']
        if "images" in self.label_contents.keys():
            self.label_contents["images"] = tr.Nav2DTransforms.dense_graph_to_minigrid_render(data, tile_size=16)

        self.compute_graph_metrics(data, features)

    def compute_graph_metrics(self, graphs, features):

        start_nodes, goal_nodes = self._compute_start_goal_indices(features)

        for i, graph in enumerate(graphs):
            start_node = int(start_nodes[i])
            goal_node = int(goal_nodes[i])
            graph, valid, solvable = graph_metrics.prepare_graph(graph, start_node, goal_node)

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

    def _compute_start_goal_indices(self, features):

        start_dim, goal_dim = self.dataset_meta['graph_feature_descriptors'].index('start'), \
                              self.dataset_meta['graph_feature_descriptors'].index('goal')

        start_nodes = features[..., start_dim].argmax(dim=-1)
        goal_nodes = features[..., goal_dim].argmax(dim=-1)

        return start_nodes, goal_nodes

    def get_images(self, indices: List[int]) -> List[np.ndarray]:
        raise NotImplementedError


class WaveCollapseBatch(Batch):

        PATTERN_COLOR_CONFIG = {
            "wall": (0, 0, 0), #black
            "empty": (255, 255, 255), #white
            }

        def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any], seeds:torch.Tensor=None):
            super().__init__(batch_meta, dataset_meta, seeds)

            self.task_structure = self.batch_meta['task_structure']
            base_dir = str(Path(__file__).resolve().parent)
            task_structure_meta_path = base_dir + f"/conf/task_structure/{self.task_structure}.yaml"
            self.task_structure_meta = OmegaConf.load(task_structure_meta_path)
            template_path = base_dir + f"/{self.task_structure_meta.template_path}"
            self.template = imageio.v2.imread(template_path)[:, :, :3]
            self.output_pattern_dim = (self.dataset_meta['data_dim'][0] - 2, self.dataset_meta['data_dim'][1] - 2)

        def generate_data(self):
            if self.seeds is None:
                self.seeds = torch.randint(0, 2 ** 32 - 1, (int(1e6),), dtype=torch.int64)
            features = []
            remove_seeds = torch.ones(self.seeds.shape, dtype=torch.bool)

            i = 0
            while len(features) < self.batch_meta['batch_size']:
                seed = self.seeds[i]
                pattern = self._run_wfc(seed)
                if pattern is None:
                    #logger.info(f"Seed {seed} failed to generate a valid pattern at iteration {i}")
                    pass
                else:
                    remove_seeds[i] = False
                    features.append(pattern)
                i += 1

            self.seeds = self.seeds[~remove_seeds]
            assert len(features) == len(self.seeds), "Number of generated patterns does not match number of seeds"

            features = np.array(features)
            features = self._pattern_to_minigrid_layout(features)
            features = tr.Nav2DTransforms.minigrid_layout_to_dense_graph(features, to_dgl=False, remove_edges=False)

            if self.dataset_meta['ensure_connected']:
                features = self._get_largest_component(features, to_dgl=True)

            features = self._place_start_and_goal(features)

            return features

        def _run_wfc(self, seed):
            util.seed_everything(seed)

            try:
                generated_pattern, stats = wfc_control.execute_wfc(tile_size=1,
                            pattern_width=self.task_structure_meta['pattern_width'],
                                        rotations=self.task_structure_meta['rotations'],
                                        output_size=self.output_pattern_dim,
                                        attempt_limit=1,
                                        output_periodic=self.task_structure_meta['output_periodic'],
                                        input_periodic=self.task_structure_meta['input_periodic'],
                                        loc_heuristic=self.task_structure_meta['loc_heuristic'],
                                        choice_heuristic=self.task_structure_meta['choice_heuristic'],
                                        backtracking=self.task_structure_meta['backtracking'],
                                        image=self.template)
            except wfc_solver.TimedOut or wfc_solver.StopEarly or wfc_solver.Contradiction:
                logger.info(f"WFC failed to generate a pattern. Outcome: {stats['outcome']}")
                return None

            return generated_pattern

        def _pattern_to_minigrid_layout(self, patterns):

            assert patterns.ndim == 4
            layouts = np.ones(patterns.shape, dtype=tr.LEVEL_INFO['dtype']) * tr.Minigrid_OBJECT_TO_IDX['empty']

            wall_ids = np.where(patterns == self.PATTERN_COLOR_CONFIG['wall'])
            layouts[wall_ids] = tr.Minigrid_OBJECT_TO_IDX['wall']
            layouts = layouts[..., 0]

            return layouts

        def _place_start_and_goal(self, graphs: List[dgl.DGLGraph]):

            for graph in graphs:
                possible_nodes = torch.where(graph.ndata['active'])[0]
                inds = torch.randperm(len(possible_nodes))[:2]
                start_node, goal_node = possible_nodes[inds]
                graph.ndata['start'][start_node] = 1
                graph.ndata['goal'][goal_node] = 1

            return graphs

        def check_pattern_validity(self, pattern):
            pass

        def _get_largest_component(self, graphs: Union[List[nx.Graph], List[dgl.DGLGraph]], to_dgl: bool = False)\
                -> Union[List[nx.Graph], List[dgl.DGLGraph]]:

            wall_graph_attr = tr.OBJECT_TO_DENSE_GRAPH_ATTRIBUTE['wall']
            for i, graph in enumerate(graphs):
                component, _, _ = graph_metrics.prepare_graph(graph)
                act = nx.get_node_attributes(component, 'active')
                g = nx.Graph()
                g.add_nodes_from(graph.nodes())
                for j in range(len(self.dataset_meta['graph_feature_descriptors'])):
                    nx.set_node_attributes(g, wall_graph_attr[j], self.dataset_meta['graph_feature_descriptors'][j])
                nx.set_node_attributes(g, act, "active")
                g.add_edges_from(component.edges(data=True))
                if to_dgl:
                    g = dgl.from_networkx(g, node_attrs=self.dataset_meta['graph_feature_descriptors'])
                graphs[i] = copy.deepcopy(g)

            return graphs

        def get_images(self, idx: List[int]) -> List[torch.Tensor]:

            data = [self.features[i] for i in idx]
            images = tr.Nav2DTransforms.dense_graph_to_minigrid_render(data, tile_size=16)
            return images


if __name__ == '__main__':
    generate_dataset()