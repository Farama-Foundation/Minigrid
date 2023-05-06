from __future__ import annotations

__package__ = "maze_representations"

import copy
import logging
import time
from collections import defaultdict

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

from util import make_grid_with_labels, DotDict
import maze_representations.data_generation.generation_algorithms.wfc_2019f.wfc.wfc_control as wfc_control
import maze_representations.data_generation.generation_algorithms.wfc_2019f.wfc.wfc_solver as wfc_solver
from maze_representations.util.distributions import compute_weights
from maze_representations.util import graph_metrics
from maze_representations.util import transforms as tr
from maze_representations.util import util as util
from maze_representations.util.multiprocessing import MockProcessPoolExecutor, Parser

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def generate_dataset(cfg: DictConfig) -> None:
    """
    Generate a dataset of gridworld navigation problems.
    """

    dataset_config = cfg.data_generation
    logger.info("\n" + OmegaConf.to_yaml(dataset_config))

    logger.info("Parsing Data Generation config to DatasetGenerator...")
    DatasetGenerator = GridNavDatasetGenerator(dataset_config,
                                               num_workers=cfg.num_cpus,
                                               multiprocessing=cfg.multiprocessing,
                                               debug_multiprocessing=cfg.debug_multiprocessing)
    if dataset_config.source_dataset is not None:
        logger.info("Obtaining dataset from source dataset")
        base_dir = str(Path(__file__).resolve().parent) + '/datasets/'
        dataset_meta_path = os.path.join(base_dir, dataset_config.source_dataset, 'dataset.meta')
        DatasetGenerator.generate_subdataset(dataset_meta_path)
    else:
        logger.info("Generating Dataset...")
        DatasetGenerator.generate_dataset()
    logger.info("Done.")


class GridNavDatasetGenerator:

    def __init__(self, dataset_config: DictConfig, num_workers=0, multiprocessing=False,
                 debug_multiprocessing=False):
        self.config = self.generate_config(dataset_config)

        if self.config.label_descriptors_config.use_seed:
            if self.config.seed is None:
                self.seed = self.config.label_descriptors_config.seed = 123456 #default seed
            else:
                self.seed = self.config.seed
            util.seed_everything(self.seed)
            logger.info(f"Using seed {self.seed}")

        self.ts_thumbnails = {}
        self.ts_batches = defaultdict(list)
        self._task_seeds = torch.randperm(int(1e7)) #10M possible seeds
        self.batch_label_offset = []

        self.dataset_meta = self.generate_dataset_metadata()
        self.batches_meta = self.generate_batches_metadata()
        self._seeds_per_batch = len(self._task_seeds) // len(self.batches_meta)
        self.save_dir = self.get_dataset_dir(self.config.dir_name)
        self.generated_batches = {}
        self.use_multiprocessing = True if num_workers != 1 and multiprocessing else False
        self.multi_processing_debug_mode = True if multiprocessing and num_workers != 1 and debug_multiprocessing \
            else False
        self.num_workers = num_workers

    def generate_config(self, dataset_config: DictConfig) -> DictConfig:
        """
        Generate a config for the dataset generator.
        """
        if not dataset_config.generate_images:
            dataset_config.label_descriptors.remove('images')
        if dataset_config.num_train_batches is None:
            if dataset_config.size_train_batch == 0:
                dataset_config.num_train_batches = 0
            else:
                dataset_config.num_train_batches = int(np.array(list(dataset_config.task_structure_split.values())).sum())
        if dataset_config.num_test_batches is None:
            if dataset_config.size_test_batch == 0:
                dataset_config.num_test_batches = 0
            else:
                dataset_config.num_test_batches = int(np.array(list(dataset_config.task_structure_split.values())).sum())

        return dataset_config

    def generate_dataset_metadata(self) -> DotDict:

        dataset_meta = DotDict({})
        dataset_meta.config = self.config

        level_info = {
            'numpy': True,
            'dtype': np.uint8,
            'shape': (self.config.gridworld_data_dim[1],
                      self.config.gridworld_data_dim[2],
                      self.config.gridworld_data_dim[0])
            }

        dataset_meta.output_file = 'dataset.meta'
        dataset_meta.unique_data = None
        dataset_meta.task_seeds = self._task_seeds
        dataset_meta.all_batches_present = False
        dataset_meta.metrics_normalised = False
        dataset_meta.ts_thumbnails = self.ts_thumbnails
        dataset_meta.ts_batches = self.ts_batches
        dataset_meta.level_info = level_info

        return dataset_meta

    def generate_batches_metadata(self):

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
                        raise NotImplementedError(f'Not supporting multiple task generation algorithms" ')
                    else:
                        algo = self.config.generating_algorithm[0]

                    batch_meta = {
                        'output_file': f'{prefix}batch_{i}.data',
                        'batch_size': batch_size,
                        'batch_id': batch_id,
                        'task_structure': task_structures[i],
                        'generating_algorithm': algo,
                        'generating_algorithm_options': [],
                        'unique_data': True,
                        }
                    batches_meta.append(batch_meta)

            else:
                logger.info(f"No {regime} batches will be generated.")
                # raise ValueError(f"Number of batches must be greater than 0.")

        batches_meta = []
        for val in all_batches_meta.values():
            batches_meta.extend(val)

        return batches_meta

    def generate_dataset(self):

        #check if folder exists
        if os.path.exists(self.save_dir):

            if not self.config.resume_generation:
                raise FileExistsError(f"Dataset directory {self.save_dir} already exists. Aborting.")
            else:
                logger.info(f"Dataset directory {self.save_dir} already exists. Resuming generation.")
            try:
                self.dataset_meta = self.load_dataset_meta()
            except FileNotFoundError:
                logger.warning(f"Dataset meta file not found in {self.save_dir}. Generating new meta file.")
                if self.existing_files:
                    raise FileExistsError(f"Dataset directory {self.save_dir} contains files but not dataset.meta. "
                                          f"Aborting.")
                self.save_dataset_meta()

        # creates folder if it does not exist.
        else:
            logger.info(f"Saving dataset in new directory {self.save_dir}")
            os.makedirs(self.save_dir)
            self.save_dataset_meta()

        logger.info(f"Batches to be generated:")
        for batch_meta in self.batches_meta:
            if batch_meta['output_file'] in self.existing_files:
                continue
            else:
                logger.info(f"Batch {batch_meta['batch_id']}")

        n_labels = 0
        seed_counter = 0
        args = []
        for i, batch_meta in enumerate(self.batches_meta):
            self.batch_label_offset.append(n_labels)
            seeds = self._task_seeds[seed_counter:seed_counter+self._seeds_per_batch]
            seed_counter += self._seeds_per_batch
            n_first_label = n_labels
            n_labels += batch_meta['batch_size']
            existing_files = self.existing_files
            if batch_meta['output_file'] in existing_files:
                logger.info(f"{batch_meta['output_file']} already exists. Will be skipped.")
                continue
            args.append((batch_meta, self.dataset_meta, seeds, n_first_label, self.save_dir))

        self.dataset_meta.size = n_labels

        if self.use_multiprocessing:
            logger.info(f"Using multiprocessing with {self.num_workers} workers.")
            multi_processing_parser = Parser(GridNavDatasetGenerator._generate_batch_data)
            if not self.multi_processing_debug_mode:
                num_cpus = self.num_workers if self.num_workers != 0 else None #None means use all available cpus
                torch.set_num_threads(1)
                # proc = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus, mp_context=multiprocessing.get_context('spawn'))
                logger.info(f"Using BoundedProcessPoolExecutor with {num_cpus} workers.")
                proc = BoundedProcessPoolExecutor(max_workers=num_cpus)
            else: #TODO: fix this
                proc = MockProcessPoolExecutor()
        else:
            proc = None

        # TODO: consider using tqdm to show progress
        if proc is not None:
            with proc as executor:
                dataset = [executor.submit(multi_processing_parser, arg) for arg in args]
                for futures_job in concurrent.futures.as_completed(dataset):
                    try:
                        batch = futures_job.result()

                        logger.info(f"Generated batch {batch.batch_meta['batch_id']}.")

                        self.update_metric_normalisation_factors(batch.label_contents)

                        if not batch.batch_meta['unique_data']:
                            self.dataset_meta.unique_data = False

                        ts = batch.batch_meta['task_structure']
                        self.ts_batches[ts].append(batch.batch_meta['output_file'])
                        if ts not in self.ts_thumbnails:
                            self.ts_thumbnails[batch.batch_meta['task_structure']] = \
                                batch.get_images([j for j in range(self.config.n_thumbnail_per_batch)])
                        del batch
                    except Exception as e:
                        logger.error(f"Error during multiprocessing. Error message: {str(e)}")
        else:
            for arg in args:
                batch = GridNavDatasetGenerator._generate_batch_data(*arg)
                logger.info(f"Generated batch {batch.batch_meta['batch_id']}.")

                self.update_metric_normalisation_factors(batch.label_contents)
                self.generated_batches[batch.batch_meta['batch_id']] = batch

        _ = self.all_batches_generated()

        if self.dataset_meta.all_batches_present \
                and self.config.normalise_metrics \
                and not self.dataset_meta.metrics_normalised:
            batch_meta_files = [f for f in self.existing_files if f.endswith('.meta') and f != 'dataset.meta']
            for file in batch_meta_files:
                loaded_data = self.load_batch_meta(file)
                self.normalise_metrics(batch_label_contents=loaded_data['label_contents'])
                GridNavDatasetGenerator.save_batch_meta(**loaded_data, save_dir=self.save_dir)
            logger.info(f"Normalised metrics for all batches.")
            self.dataset_meta.metrics_normalised = True

        if self.dataset_meta.unique_data is None:
            if self.config.check_unique:
                logger.info(f"Checking for duplicate data across entire dataset.")
                batches = [self.load_batch(f)[0] for f in self.existing_files if f.endswith('.data')]
                self.dataset_meta.unique_data = self.check_unique(batches)
            else:
                logger.info(f"Not checking for duplicate data across entire dataset, "
                               f"but data within each batch is unique.")

        self.save_dataset_meta() #Save again in case it was updated during generation
        self.save_layout_thumbnail(n_per_batch=self.config.n_thumbnail_per_batch)

    def generate_subdataset(self, original_dataset_meta_path):

        def get_data(batch_ids, batch_size):
            data = []
            for file in batch_ids:
                file = file + '.data'
                # TODO: Needed to load older versions of the dataset due to a bug in the original code
                entry, entry_type = self.load_batch(file, original_dataset_dir)
                if entry_type == 'pickle':
                    graphs = entry['data']
                    labels = entry['labels']
                    batch_meta = entry['batch_meta']
                    label_contents = entry['label_contents']
                elif entry_type == 'dgl':
                    graphs = entry[0]
                    labels = entry[1]
                    file = file + '.meta'
                    entry = self.load_batch_meta(file, original_dataset_dir)
                    batch_meta = entry['batch_meta']
                    label_contents = entry['label_contents']
                else:
                    raise ValueError(f"Unknown entry type {entry_type}.")
                graphs = graphs[:batch_size]
                labels = labels[:batch_size]
                for key in label_contents:
                    label_contents[key] = label_contents[key][:batch_size]
                batch_meta['batch_size'] = batch_size
                data.append(DotDict({'features':graphs,
                                     'label_ids':labels,
                                     'batch_meta':batch_meta,
                                     'label_contents':label_contents}))
                data[-1].batch_meta = data[-1].batch_meta.to_dict()
                data[-1].label_contents = data[-1].label_contents.to_dict()
            return data

        original_dataset_meta = self.load_dataset_meta(original_dataset_meta_path, update_config=False)
        # assert not original_dataset_meta.metrics_normalised, "Original dataset must NOT have normalised metrics."
        original_dataset_dir = os.path.dirname(original_dataset_meta_path)

        batch_data = []
        batch_data.extend(get_data(self.config.train_batch_ids, self.config.size_train_batch))
        batch_data.extend(get_data(self.config.test_batch_ids, self.config.size_test_batch))

        label_id_count = 0
        for i, b_data in enumerate(batch_data):
            assert b_data.batch_meta['task_structure'] == self.batches_meta[i]['task_structure']
            assert b_data.batch_meta['batch_size'] == self.batches_meta[i]['batch_size']
            assert b_data.batch_meta['unique_data']
            b_data.label_ids = torch.arange(label_id_count, label_id_count + b_data.batch_meta['batch_size']).to(torch.int64)
            label_id_count += b_data.batch_meta['batch_size']
            if self.config.task_type == 'cave_escape' and original_dataset_meta['task_type'] == 'navigate_to_goal':
                logger.info(f"Converting batch {i}/{len(batch_data)} to cave escape task.")
                seeds = b_data.label_contents['seed']
                b_data.features, b_data.label_contents = self._convert_batch_data(b_data, self.batches_meta[i])
                b_data.label_contents['seed'] = seeds
            data = [b_data.features[i] for i in range(self.config.n_thumbnail_per_batch)]
            images = tr.Nav2DTransforms.dense_graph_to_minigrid_render(data, tile_size=16, level_info=self.dataset_meta.level_info)
            self.ts_thumbnails[b_data.batch_meta['task_structure']] = images
            self.update_metric_normalisation_factors(b_data.label_contents)

        for b_data in batch_data:
            self.normalise_metrics(batch_label_contents=b_data.label_contents)
            self.save_batch(b_data, self.save_dir)
        logger.info(f"Normalised metrics for all batches.")
        self.dataset_meta.metrics_normalised = True

        self.save_dataset_meta()
        self.save_layout_thumbnail(n_per_batch=self.config.n_thumbnail_per_batch)

    def _convert_batch_data(self, batch_data, new_batch_meta):
        time_batch_start = time.perf_counter()
        batch = MinigridToCaveEscapeBatch(batch_meta=new_batch_meta, dataset_meta=self.dataset_meta,
                                          seeds=batch_data.label_contents['seed'])
        batch_features, _, batch_label_contents = batch.generate_batch(batch_data)
        time_batch_end = time.perf_counter()
        batch_generation_time = time_batch_end - time_batch_start
        logger.info(f"Batch {new_batch_meta['batch_id']}, Task structure: {new_batch_meta['task_structure']} generated in "
                    f"{batch_generation_time:.2f} seconds. "
                    f"Average time per sample: {batch_generation_time / len(batch_features):.2f} seconds.")

        return batch_features, batch_label_contents

    @staticmethod
    def _generate_batch_data(batch_meta, dataset_meta, seeds, batch_label_offset, save_dir):
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

        GridNavDatasetGenerator.save_batch(batch, save_dir)
        return batch

    def update_metric_normalisation_factors(self, label_contents: Dict):
        """
        Update the metric normalisation factors for the dataset based on the latest batch.
        """

        maximums = ["shortest_path", "resistance", "navigable_nodes"]
        for key in maximums:
            batch_metric = label_contents[key]
            if isinstance(batch_metric, list) or isinstance(batch_metric, tuple):
                batch_metric = torch.tensor(batch_metric)
            elif isinstance(batch_metric, np.ndarray):
                batch_metric = torch.from_numpy(batch_metric)
            elif isinstance(batch_metric, torch.Tensor):
                pass
            else:
                raise TypeError(f"Metric must be a list, tuple, numpy array or torch tensor. got: {type(batch_metric)}")
            metric_max = batch_metric.amax().item() * self.dataset_meta.config.label_descriptors_config[key]['max']
            if self.dataset_meta.config.label_descriptors_config[key]['normalisation_factor'] is None \
                    or metric_max > self.dataset_meta.config.label_descriptors_config[key]['normalisation_factor']:
                self.dataset_meta.config.label_descriptors_config[key]['normalisation_factor'] = metric_max

    def normalise_metrics(self, batch_label_contents=None, batches=None):
        """
        Normalize metrics either across batches or for a specific batch_label_contents.
        Either batch_label_contents or batches must be provided but not both.

        :param batch_label_contents: A dictionary containing the label contents for a specific batch (optional).
        :param batches: A list of batches (optional).
        """

        assert (batch_label_contents is None) != (batches is None), "Either batch_label_contents or batches must be provided, but not both."

        maximums = ["shortest_path", "resistance", "navigable_nodes"]
        for key in maximums:
            # Use 1. perform (and possibly compute) the normalisation across batches
            if batches is not None:
                if self.dataset_meta.config.label_descriptors_config[key]['normalisation_factor'] is None:
                    stack = []
                    for batch in batches:
                        stack.append(batch.label_contents[key])
                    stack = torch.cat(stack)
                    self.dataset_meta.config.label_descriptors_config[key]['normalisation_factor'] = \
                        float(stack.amax() * self.dataset_meta.config.label_descriptors_config[key]['max'])
                for batch in batches:
                    batch.label_contents[key] = batch.label_contents.to(torch.float) \
                        / self.dataset_meta.config.label_descriptors_config[key]['normalisation_factor']
            # Use 2. perform the normalisation across a specific batch_label_contents, using precomputed factors
            else:
                assert self.dataset_meta.config.label_descriptors_config[key]['normalisation_factor'] is not None, \
                    "Normalisation factor for metric {} not computed.".format(key)
                batch_label_contents[key] = batch_label_contents[key].to(torch.float) \
                                            / self.dataset_meta.config.label_descriptors_config[key]['normalisation_factor']

    def all_batches_generated(self):
        """
        Called to check if all batches have been generated.
        """

        all_generated = True
        existing_files = self.existing_files
        for batch_meta in self.batches_meta:
            if batch_meta['output_file'] not in existing_files:
                logger.warning(f"Missing {batch_meta['output_file']}.")
                all_generated = False

        if all_generated:
            logger.info("All batches generated successfully.")

        self.dataset_meta.all_batches_present = all_generated

        return all_generated

    def get_dataset_dir(self, dir_name=None):
        base_dir = str(Path(__file__).resolve().parent) + '/datasets/'

        if dir_name is None:
            task_structures = '-'.join(sorted(self.config.task_structure_split.keys()))
            dir_name = f"ts={task_structures}-x={self.dataset_meta.data_type}-s={self.config.size}" \
                                f"-d={self.config.gridworld_data_dim[-1]}-gf={len(self.config.graph_feature_descriptors)}" \
                                f"-enc={self.config.encoding}"

            if len(base_dir + dir_name + '/') > 260:
                task_structures = 'many'
                dir_name = f"ts={task_structures}-x={self.dataset_meta.data_type}-s={self.config.size}" \
                           f"-d={self.config.gridworld_data_dim[-1]}-gf={len(self.config.graph_feature_descriptors)}" \
                           f"-enc={self.config.encoding}"

                logger.warning(f"Dataset directory name too long. Using {dir_name} instead.")

        path = base_dir + dir_name + '/'

        if len(path) > 260:
            raise ValueError("Dataset dir path too long. Please use a shorter path.")

        return path

    @staticmethod
    def save_batch(batch: Union[Batch, Dict, DotDict], save_dir):
        """
        Save a batch to disk.
        :param save_dir:
        :param batch: May be provided as a Batch object or as a dictionary containing only the data to be saved.
        """

        filename = batch.batch_meta['output_file']
        filepath = save_dir + filename
        if isinstance(batch.label_ids, np.ndarray):
            batch.label_ids = torch.tensor(batch.label_ids)

        # need to save (data, labels) and (label_contents, metadata) in 2 separate files because of limitations of
        # save_graphs()
        logger.info(f"Saving Batch {batch.batch_meta['batch_id']} to {filepath}.")
        dgl.data.utils.save_graphs(filepath, batch.features, {'labels': batch.label_ids})
        GridNavDatasetGenerator.save_batch_extra_data(batch, save_dir)
        GridNavDatasetGenerator.save_batch_meta(batch.label_contents, batch.batch_meta, save_dir)
        # entry = {'data': batch.features, 'labels': batch.label_ids,
        #          'label_contents': batch.label_contents, 'batch_meta': batch.batch_meta}
        # with open(filepath, 'wb') as f:
        #     pickle.dump(entry, f)
        logger.info(f"Saved Batch {batch.batch_meta['batch_id']} to {filepath}.")

    @staticmethod
    def save_batch_meta(label_contents, batch_meta, save_dir):
        filename = batch_meta['output_file'] + '.meta'
        filepath = save_dir + filename
        logger.info(f"Saving Batch {batch_meta['batch_id']} Metadata to {filepath}.")
        entry = {'label_contents': label_contents, 'batch_meta': batch_meta}
        extra_data = {}
        if "edge_graphs" in label_contents:
            extra_data["edge_graphs"] = label_contents.pop("edge_graphs")
            filename_extra = batch_meta['output_file'] + '.dgl.extra'
            filepath_extra = save_dir + filename_extra
            logger.info(f"Saving Batch {batch_meta['batch_id']} Extra Data to {filepath_extra}.")
            dgl.data.utils.save_graphs(filepath_extra, extra_data["edge_graphs"])
        with open(filepath, 'wb') as f:
            pickle.dump(entry, f)

    @staticmethod
    def save_batch_extra_data(batch, save_dir):
        extra_data = {}
        if "edge_graphs" in batch.label_contents:
            extra_data["edge_graphs"] = batch.label_contents.pop("edge_graphs")
            filename_extra = batch.batch_meta['output_file'] + '.dgl.extra'
            filepath_extra = save_dir + filename_extra

            extra_graphs = []
            extra_labels = {}
            for key in extra_data["edge_graphs"]:
                extra_graphs.extend(extra_data["edge_graphs"][key])
                extra_labels[key] = batch.label_ids.clone()

            logger.info(f"Saving Batch {batch.batch_meta['batch_id']} Extra Data to {filepath_extra}.")
            dgl.data.utils.save_graphs(filepath_extra, extra_graphs, extra_labels)
        else:
            pass

    def load_batch_extra_data(self, filename, dir=None):
        if dir is None:
            dir = self.save_dir
        filepath = os.path.join(dir, filename)
        try:
            entry = dgl.load_graphs(filepath)
            extra_data = util.assemble_extra_data(entry)
            return extra_data
        except FileNotFoundError:
            logger.warning(f"Could not find {filepath}.")
            return None

    def load_batch(self, filename, dir=None):
        if dir is None:
            dir = self.save_dir
        filepath = os.path.join(dir, filename)
        try:
            entry = dgl.load_graphs(filepath)
            entry_type = 'dgl'
        # TODO: needed for backwards compatibility. Remove after a while.
        except Exception as e:
            logger.error(f"Error loading {filepath} with DGL."
                         f"Message: {e}."
                         f"Trying to load with pickle.")
            with open(filepath, 'rb') as f:
                entry = pickle.load(f)
                entry_type = 'pickle'
            logger.info(f"Successfuly Loaded {filepath} with pickle.")
        return entry, entry_type

    def load_batch_meta(self, filename, dir=None) -> Dict[str, Any]:
        '''
        Load the metadata of a batch.
        :param filename:
        :return: {'label_contents': batch_label_contents, 'batch_meta': batch_meta}
        '''
        if dir is None:
            dir = self.save_dir
        filepath = os.path.join(dir, filename)
        with open(filepath, 'rb') as f:
            entry = pickle.load(f)
        logger.info(f"Loaded Batch {entry['batch_meta']['batch_id']} Metadata from {filepath}.")
        return entry

    def save_dataset_meta(self):

        entry = self.dataset_meta

        filename = self.dataset_meta.output_file
        filepath = self.save_dir + filename
        logger.info(f"Saving dataset metadata to {filepath}.")
        with open(filepath, 'wb') as f:
            pickle.dump(entry, f)

    def load_dataset_meta(self, dataset_meta_path=None, update_config=True):

        if dataset_meta_path is None:
            path = os.path.join(self.save_dir, self.dataset_meta.output_file)
        else:
            path = dataset_meta_path
        with open(path, "rb") as infile:
            dataset_meta = pickle.load(infile, encoding="latin1")

        if update_config:
            self.config = dataset_meta.config
            # Necessary to ensure that the same seeds are used if resuming generation
            self._task_seeds = dataset_meta.task_seeds
            # Necessary to ensure we keep the existing thumbnails if resuming generation
            self.ts_thumbnails = dataset_meta.ts_thumbnails

        return dataset_meta

    def save_layout_thumbnail(self, n_per_batch=10):
        filename = 'layouts.png'

        images = []
        task_structures = []
        if not self.ts_thumbnails:
            for batch in self.generated_batches.values():
                label_content = batch.label_contents

                #Skip repeated values
                if batch.batch_meta['task_structure'] in task_structures:
                    continue

                task_structures.extend(label_content['task_structure'][:n_per_batch])
                if 'images' in label_content:
                    img = label_content['images'][:n_per_batch]
                else:
                    img = batch.get_images([j for j in range(n_per_batch)])
                if n_per_batch > len(img):
                    logger.warning(f"Batch {batch.batch_meta['batch_id']} has less than {n_per_batch} images."
                                   f"{filename} may not be generated correctly")
                images.extend(img)
        else:
            for ts, img in self.ts_thumbnails.items():
                task_structures.extend([ts] * len(img))
                images.extend(img)

        task_structures = ['']*len(images)
        path = os.path.join(self.save_dir, filename)
        images = make_grid_with_labels(images, task_structures, nrow=n_per_batch, normalize=True, limit=None,
                                       channels_first=True)
        logger.info(f"Saving layout thumbnails to {path}.")
        vutils.save_image(images, path)

    def check_unique(self, batches: Union[List[Batch], List[Tuple[torch.Tensor, torch.Tensor]]]):

        all_data = []
        for batch in batches:
            if isinstance(batch, Batch):
                all_data.extend(batch.features)
            elif isinstance(batch, tuple):
                all_data.extend(batch[0])
            else:
                raise TypeError(f"Batch type {type(batch)} not recognised.")
        features, _ = util.get_node_features(all_data, device=None) # TODO: add device?
        if not util.check_unique(features).all():
            logger.warning(f"There are duplicate features in the dataset.")
            return False
        else:
            return True

    @property
    def existing_files(self) -> List[str]:
        return [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]


class BatchGenerator:

    def __new__(cls, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any], seeds:torch.Tensor=None):

        if batch_meta['generating_algorithm'] == 'wave_function_collapse':
            instance = super().__new__(CaveEscapeWaveCollapseBatch)
        else:
            raise KeyError("Task Structure was not recognised")

        instance.__init__(batch_meta, dataset_meta, seeds=seeds)

        return instance

    def __init__(self):
        raise RuntimeError(f"{type(self)} is a Class Factory. Assign it to a variable. ")


class Batch:

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: DictConfig, seeds:torch.Tensor=None):
        self.seeds = seeds
        self.batch_meta = batch_meta
        self.dataset_meta = dataset_meta
        self.data_type = dataset_meta.config.data_type
        self.encoding = dataset_meta.config.encoding
        self.label_ids = np.arange(self.batch_meta['batch_size'])
        self.label_contents = dict.fromkeys(self.dataset_meta.config.label_descriptors)
        self.features = None
        self.images = None

    def generate_batch(self):

        graphs, extra = self.generate_data()
        if isinstance(graphs, dgl.DGLGraph) or isinstance(graphs[0], dgl.DGLGraph):
            features, _ = util.get_node_features(graphs, device=None) # TODO: add device?
        else:
            features = graphs

        if not util.check_unique(features).all():
            logger.warning(f"Batch {self.batch_meta['batch_id']} generated duplicate features.")
            self.batch_meta['unique_data'] = False

        if self.data_type == 'graph':
            if self.encoding == 'dense':
                if not (isinstance(graphs, dgl.DGLGraph) or isinstance(graphs[0], dgl.DGLGraph)):
                    raise TypeError("Output of generate_data() must be a DGLGraph or list of DGLGraphs. For dense "
                                    "encoding")
            else:
                raise NotImplementedError(f"Encoding {self.encoding} not implemented.")
            self.generate_label_contents(graphs, extra_data=extra)
        else:
            raise NotImplementedError(f"Data type {self.data_type} not implemented.")

        self.features = graphs
        return self.features, self.label_ids, self.label_contents

    def generate_data(self) -> Tuple[List[dgl.DGLGraph], Dict[str, Any]]:
        raise NotImplementedError

    def generate_label_contents(self, data, extra_data=None):

        for key in self.label_contents.keys():
            self.label_contents[key] = []

        self.label_contents["seed"] = self.seeds if self.seeds is not None else [None] * self.batch_meta['batch_size']
        self.label_contents["task_structure"] = [self.batch_meta["task_structure"]] * self.batch_meta['batch_size']
        self.label_contents["generating_algorithm"] = [self.batch_meta["generating_algorithm"]] * self.batch_meta['batch_size']
        self.label_contents['minigrid'] = tr.Nav2DTransforms.dense_graph_to_minigrid(data, level_info=self.dataset_meta.level_info)
        if "images" in self.label_contents.keys():
            self.label_contents["images"] = tr.Nav2DTransforms.dense_graph_to_minigrid_render(data, tile_size=16, level_info=self.dataset_meta.level_info)

        self.compute_graph_metrics(extra_data["edge_graphs"]["navigable"])

        if extra_data is not None:
            for key, value in extra_data.items():
                self.label_contents[key] = value

    def compute_graph_metrics(self, graphs):

        metrics = graph_metrics.compute_metrics(graphs)

        start_nodes, goal_nodes = self._compute_start_goal_indices(graphs)
        metrics["optimal_trajectories"] = []
        for i, graph in enumerate(graphs):
            g_nx = util.dgl_to_nx(graph)
            shortest_paths = graph_metrics.shortest_paths(g_nx, start_nodes[i], goal_nodes[i], num_paths=1)
            metrics["optimal_trajectories"].append(shortest_paths)

        for key, value in metrics.items():
            self.label_contents[key] = value

    def _compute_start_goal_indices(self, graphs: Union[dgl.DGLGraph, List[dgl.DGLGraph]]):

        feature_tensor, feat_labels = util.get_node_features(graphs)

        start_dim, goal_dim = feat_labels.index('start'), feat_labels.index('goal')

        start_nodes = feature_tensor[..., start_dim].argmax(dim=-1).tolist()
        goal_nodes = feature_tensor[..., goal_dim].argmax(dim=-1).tolist()

        return start_nodes, goal_nodes

    def get_images(self, indices: List[int]) -> List[np.ndarray]:
        raise NotImplementedError


class CaveEscapeWaveCollapseBatch(Batch):

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
            # with open(template_path, "r") as f:
            #     self.template = imageio.v2.imread(f)[:, :, :3]
            self.template = imageio.v2.imread(template_path)[:, :, :3]
            self.output_pattern_dim = (self.dataset_meta.level_info['shape'][0] - 2, self.dataset_meta.level_info['shape'][1] - 2)

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

            stage1_edge_config = {}
            stage2_edge_config = {}
            for k, v in self.dataset_meta.config.graph_edge_descriptors.items():
                if k == 'navigable':
                    stage1_edge_config[k] = v
                else:
                    stage2_edge_config[k] = v

            features, edge_graphs = tr.Nav2DTransforms.minigrid_layout_to_dense_graph(features,
                                                                         to_dgl=False,
                                                                         remove_border=False,
                                                                         node_attr=self.dataset_meta.config.graph_feature_descriptors,
                                                                         edge_config=stage1_edge_config,
                                                                         to_dgl_edge_g=True)

            if self.dataset_meta.config.ensure_connected:
                features = self._get_largest_component(features, to_dgl=False)

            extra = {}
            if self.dataset_meta.config.task_type == "navigate_to_goal":
                features, _ = self._place_start_and_goal_random(features)
            elif self.dataset_meta.config.task_type == "cave_escape":
                extra['shortest_path_dist'] = self._place_goal_random(features)
                extra['probs_moss'] = self._place_moss_cave_escape(features, extra['shortest_path_dist'])
                extra['probs_lava'] = self._place_lava_cave_escape(features, extra['shortest_path_dist'])
                extra['alternate_start_locations'] = self._place_start_cave_escape(features, extra['shortest_path_dist'])
                extra['edge_graphs'] = self._add_edges(features, stage2_edge_config, edge_graphs=edge_graphs)
                features = [util.nx_to_dgl(f, enable_warnings=False) for f in features]
                self._update_graph_features(extra['edge_graphs'], features)

            return features, extra

        @staticmethod
        def _update_graph_features(graphs:Dict[str, List[dgl.DGLGraph]], reference_graphs:List[dgl.DGLGraph]):
            reference_graphs = dgl.batch(reference_graphs)

            for key in graphs:
                graphs[key] = dgl.batch(graphs[key])
                for feat in reference_graphs.ndata:
                    graphs[key].ndata[feat] = reference_graphs.ndata[feat]
                graphs[key] = dgl.unbatch(graphs[key])

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
            layouts = np.ones(patterns.shape, dtype=self.dataset_meta.level_info['dtype']) * tr.Minigrid_OBJECT_TO_IDX['empty']

            wall_ids = np.where(patterns == self.PATTERN_COLOR_CONFIG['wall'])
            layouts[wall_ids] = tr.Minigrid_OBJECT_TO_IDX['wall']
            layouts = layouts[..., 0]

            return layouts

        @staticmethod
        def _place_start_and_goal_random(graphs: List[dgl.DGLGraph]):

            node_set = 'navigable'

            for graph in graphs:
                possible_nodes = torch.where(graph.ndata[node_set])[0]
                inds = torch.randperm(len(possible_nodes))[:2]
                start_node, goal_node = possible_nodes[inds]
                graph.ndata['start'][start_node] = 1
                graph.ndata['goal'][goal_node] = 1

            return graphs, None

        @staticmethod
        def _place_goal_random(graphs: List[nx.Graph]):

            node_set = 'empty'

            spl = []
            for graph in graphs:
                possible_nodes = [n for n in graph.nodes() if graph.nodes[n][node_set]==1.0]
                ind = torch.randperm(len(possible_nodes))[0].item()
                goal_node = possible_nodes[ind]
                graph.nodes[goal_node]['goal'] = 1.0
                graph.nodes[goal_node][node_set] = 0.0
                spl.append(dict(nx.single_target_shortest_path_length(graph, goal_node)))

            return spl

        def _place_moss_cave_escape(self, graphs: List[nx.Graph], spl):

            assert len(graphs) == len(spl), "Number of graphs and shortest path lengths do not match"

            params = self.dataset_meta.config.moss_distribution_params
            assert params.nodes[0] == 'empty' and len(params.nodes) == 1, "Only implemented for moss on empty nodes"
            node_set = params.nodes[0]

            probs = []
            for m, graph in enumerate(graphs):
                if hasattr(params.fraction, '__iter__'):
                    fraction = np.random.uniform(params.fraction[0], params.fraction[1])
                else:
                    fraction = params.fraction
                possible_nodes = [n for n in graph.nodes() if graph.nodes[n][node_set]==1.0]
                num_sampled = int(fraction * len(possible_nodes))
                shortest_path_lengths = spl[m]
                shortest_path_lengths = {k: v for k, v in shortest_path_lengths.items() if k in possible_nodes}
                scores = -np.array(list(shortest_path_lengths.values()))
                weights = compute_weights(scores, params)
                # sample moss according to weights
                sampled_inds = np.random.choice(len(possible_nodes), size=num_sampled, replace=False, p=weights)
                sampled_nodes = [list(shortest_path_lengths.keys())[i] for i in sampled_inds]
                for node in sampled_nodes:
                    graph.nodes[node]['moss'] = 1.0
                    for node_subset_ in params.nodes:
                        graph.nodes[node][node_subset_] = 0.0
                pp = {}
                for node in graph.nodes():
                    if node not in possible_nodes:
                        pp[node] = 0.0
                    else:
                        pp[node] = weights[list(shortest_path_lengths.keys()).index(node)]
                probs.append(pp)

            return probs

        def _place_lava_cave_escape(self, graphs: List[nx.Graph], spl):

            assert len(graphs) == len(spl), "Number of graphs and shortest path lengths do not match"

            params = self.dataset_meta.config.lava_distribution_params
            assert params.nodes[0] == 'wall' and len(params.nodes) == 1, "Only implemented for lava on wall nodes"
            node_set = params.nodes[0]
            grid_size = (self.dataset_meta.config.gridworld_data_dim[1] - 2, self.dataset_meta.config.gridworld_data_dim[2] - 2)
            depth = params.get('sampling_depth', 3)

            probs = []
            for m, graph in enumerate(graphs):
                if hasattr(params.fraction, '__iter__'):
                    fraction = np.random.uniform(params.fraction[0], params.fraction[1])
                else:
                    fraction = params.fraction
                non_nav_nodes = [n for n in graph.nodes() if graph.nodes[n][node_set]==1.0]
                nav_nodes = [n for n in graph.nodes() if graph.nodes[n]['navigable']==1.0]
                shortest_path_lengths_nav = spl[m]
                shortest_path_lengths_nav = {k: v for k, v in shortest_path_lengths_nav.items() if k in nav_nodes}
                shortest_path_lengths = graph_metrics.get_non_nav_spl(non_nav_nodes, shortest_path_lengths_nav, grid_size, depth)
                num_sampled = int(fraction * len(shortest_path_lengths))
                scores = np.array(list(shortest_path_lengths.values()))
                weights = compute_weights(scores, params)
                sampled_inds = np.random.choice(len(shortest_path_lengths), size=num_sampled, replace=False, p=weights)
                sampled_nodes = [list(shortest_path_lengths.keys())[i] for i in sampled_inds]
                for node in sampled_nodes:
                    graph.nodes[node]['lava'] = 1.0
                    for node_subset_ in params.nodes:
                        graph.nodes[node][node_subset_] = 0.0
                pp = {}
                for node in graph.nodes():
                    if node not in shortest_path_lengths.keys():
                        pp[node] = 0.0
                    else:
                        pp[node] = weights[list(shortest_path_lengths.keys()).index(node)]
                probs.append(pp)

            return probs

        @staticmethod
        def _place_start_cave_escape(graphs: List[nx.Graph],spl):

            assert len(graphs) == len(spl), "Number of graphs and shortest path lengths do not match"

            node_set = 'navigable'

            node_subset = ['empty', 'moss']

            # sorted_weights_moss, sorted_inds_moss = torch.sort(weights_moss, descending=True)
            # sorted_weights_lava, sorted_inds_lava = torch.sort(weights_lava, descending=False)
            #
            # diff = (sorted_weights_moss - sorted_weights_lava).abs()
            # lowest_diff_ind = torch.where(diff == diff.min())[0][0]
            # start_node_inds = sorted_inds_moss[lowest_diff_ind]

            all_possible_starts = []
            for m, graph in enumerate(graphs):
                possible_nodes = [n for n in graph.nodes() if graph.nodes[n][node_set]==1.0 and graph.nodes[n]['goal']==0.0]
                shortest_path_lengths = spl[m]
                shortest_path_lengths = {k: v for k, v in shortest_path_lengths.items() if k in possible_nodes}
                spl_median = int(np.median(list(shortest_path_lengths.values())))
                start_candidates = [k for k, v in shortest_path_lengths.items() if v == spl_median]
                all_possible_starts.append(start_candidates)
                start_node_ind = np.random.choice(len(start_candidates))
                start_node = start_candidates[start_node_ind]
                graph.nodes[start_node]['start'] = 1.0
                for node_subset_ in node_subset:
                    graph.nodes[start_node][node_subset_] = 0.0

            return all_possible_starts

        def _add_edges(self, graphs: List[nx.Graph], edge_config: Union[Dict, DictConfig],
                       edge_graphs:Dict[str, List[dgl.DGLGraph]]=None)\
                ->Dict[str, List[dgl.DGLGraph]]:

            graph_features = self.dataset_meta.config.graph_feature_descriptors
            if edge_graphs is None:
                edge_graphs = defaultdict(list)
            else:
                edge_graphs = defaultdict(list, edge_graphs)
            dim_grid = tuple(d-2 for d in self.dataset_meta.level_info['shape'][0:2])

            for m, g in enumerate(graphs):
                edge_layers = tr.Nav2DTransforms.get_edge_layers(g, edge_config, graph_features, dim_grid)
                for edge_n, edge_g in edge_layers.items():
                    g.add_edges_from(edge_g.edges(data=True), label=edge_n)  # TODO: why data=True
                graphs[m] = g
                for _edge_g_type in edge_layers:
                    edge_graph = nx.convert_node_labels_to_integers(edge_layers[_edge_g_type])
                    edge_graph = dgl.from_networkx(edge_graph, node_attrs=self.dataset_meta.config.graph_feature_descriptors)
                    edge_graphs[_edge_g_type].append(edge_graph)

            return edge_graphs

        def _get_largest_component(self, graphs: Union[List[nx.Graph], List[dgl.DGLGraph]], to_dgl: bool = False)\
                -> Union[List[nx.Graph], List[dgl.DGLGraph]]:

            wall_graph_attr = tr.OBJECT_TO_DENSE_GRAPH_ATTRIBUTE['wall']
            for i, graph in enumerate(graphs):
                component, _, _ = graph_metrics.prepare_graph(graph)
                for node in graph.nodes():
                    if node not in component.nodes():
                        for feat in graph.nodes[node]:
                            if feat in wall_graph_attr:
                                graph.nodes[node][feat] = 1.0
                            else:
                                graph.nodes[node][feat] = 0.0
                g = nx.Graph()
                g.add_nodes_from(graph.nodes(data=True))
                g.add_edges_from(component.edges(data=True))
                if to_dgl:
                    g = nx.convert_node_labels_to_integers(g)
                    g = dgl.from_networkx(g, node_attrs=self.dataset_meta.config.graph_feature_descriptors)
                graphs[i] = copy.deepcopy(g)

            return graphs

        def check_pattern_validity(self, pattern):
            pass

        def get_images(self, idx: List[int]) -> List[torch.Tensor]:

            data = [self.features[i] for i in idx]
            images = tr.Nav2DTransforms.dense_graph_to_minigrid_render(data, tile_size=16, level_info=self.dataset_meta.level_info)
            return images


class MinigridToCaveEscapeBatch(CaveEscapeWaveCollapseBatch):
    PATTERN_COLOR_CONFIG = {
        "wall" :(0, 0, 0),  # black
        "empty":(255, 255, 255),  # white
        }

    def __init__(self, batch_meta: Dict[str, Any], dataset_meta: Dict[str, Any], seeds: torch.Tensor = None):
        super().__init__(batch_meta, dataset_meta, seeds)

    def generate_batch(self, batch_data: DotDict):

        graphs, extra = self.generate_data(batch_data)
        feature_tensor, _ = util.get_node_features(graphs, device=None) # TODO: add device?

        if not util.check_unique(feature_tensor).all():
            logger.warning(f"Batch {self.batch_meta['batch_id']} generated duplicate features.")
            self.batch_meta['unique_data'] = False

        self.generate_label_contents(graphs, extra_data=extra)

        self.features = graphs
        return self.features, self.label_ids, self.label_contents

    def generate_data(self, batch_data: DotDict):

        new_g, extra = self.minigrid_to_cave_escape(batch_data.features, level_info=self.dataset_meta.level_info, replace_start=True, replace_goal=True)

        stage1_edge_config = {}
        stage2_edge_config = {}
        for k, v in self.dataset_meta.config.graph_edge_descriptors.items():
            if k == 'navigable':
                stage1_edge_config[k] = v
            else:
                stage2_edge_config[k] = v

        edge_graphs = {'navigable': [new_g[i] for i in range(len(new_g))]}
        for edge_type in edge_graphs:
            for m, edge_graph in enumerate(edge_graphs[edge_type]):
                if isinstance(edge_graph, nx.Graph):
                    edge_graphs[edge_type][m] = nx.convert_node_labels_to_integers(edge_graphs[edge_type][m])
                    edge_graphs[edge_type][m] = dgl.from_networkx(edge_graphs[edge_type][m],
                                                                  node_attrs=self.dataset_meta.config.graph_feature_descriptors)

        extra['edge_graphs'] = self._add_edges(new_g, stage2_edge_config, edge_graphs=edge_graphs)
        new_g = [nx.convert_node_labels_to_integers(g) for g in new_g]
        new_g = [dgl.from_networkx(g, node_attrs=self.dataset_meta.config.graph_feature_descriptors) for g in
                 new_g]
        self._update_graph_features(extra['edge_graphs'], new_g)
        return new_g, extra

    def minigrid_to_cave_escape(self, graphs: List[dgl.DGLGraph], level_info: Dict[str, Any], replace_start: bool = False,
                                replace_goal: bool = False) -> Tuple[List[nx.Graph], Dict[str, Any]]:
        # Super weird function split but it's the easiest way to reuse this method for conversion of individual levels
        f = dgl.batch(graphs)
        if f.ndata.get('navigable') is None:
            nav_f = f.ndata['active'].to(torch.float)
        else:
            nav_f = f.ndata['navigable'].to(torch.float)
        non_nav_f = torch.zeros_like(nav_f)
        non_nav_f[nav_f == 0.0] = 1.0

        new_f = {
            'navigable': nav_f.clone(),
            'non_navigable': non_nav_f.clone(),
            'empty': nav_f.clone(),
            'wall': non_nav_f.clone(),
            'start': torch.zeros_like(nav_f),
            'goal': torch.zeros_like(nav_f),
            'lava': torch.zeros_like(nav_f),
            'moss': torch.zeros_like(nav_f)
        }

        if not replace_start:
            new_f['start'] = f.ndata['start'].to(torch.float)
        if not replace_goal:
            new_f['goal'] = f.ndata['goal'].to(torch.float)

        new_g = f.clone()
        attributes = list(new_f.keys())
        for key in attributes:
            new_g.ndata[key] = new_f[key]

        if new_g.ndata.get('active') is not None:
            new_g.ndata.pop('active')

        new_g = tr.Nav2DTransforms.graph_to_grid_graph(new_g, level_info=level_info)

        extra = {}
        if replace_goal:
            extra['shortest_path_dist'] = self._place_goal_random(new_g)
        else:
            spl = []
            for g in new_g:
                goal_node = [n for n in g.nodes if g.nodes[n]['goal'] == 1.0][0]
                spl.append(dict(nx.single_target_shortest_path_length(g, goal_node)))
            extra['shortest_path_dist'] = spl
        extra['probs_moss'] = self._place_moss_cave_escape(new_g, extra['shortest_path_dist'])
        extra['probs_lava'] = self._place_lava_cave_escape(new_g, extra['shortest_path_dist'])
        if replace_start:
            extra['alternate_start_locations'] = self._place_start_cave_escape(new_g, extra['shortest_path_dist'])

        return new_g, extra



if __name__ == '__main__':
    generate_dataset()