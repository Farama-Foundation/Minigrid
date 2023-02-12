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

    def generate_dataset_metadata(self):

        dataset_meta = OmegaConf.create()
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
                batches = [self.load_batch(f) for f in self.existing_files if f.endswith('.data')]
                self.dataset_meta.unique_data = self.check_unique(batches)
            else:
                logger.info(f"Not checking for duplicate data across entire dataset, "
                               f"but data within each batch is unique.")

        self.save_dataset_meta() #Save again in case it was updated during generation
        self.save_layout_thumbnail(n_per_batch=self.config.n_thumbnail_per_batch)

    def generate_subdataset(self, original_dataset_meta_path):

        original_dataset_meta = self.load_dataset_meta(original_dataset_meta_path, update_config=False)
        assert not original_dataset_meta.metrics_normalised, "Original dataset must NOT have normalised metrics."
        original_dataset_dir = os.path.dirname(original_dataset_meta_path)

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
        with open(filepath, 'wb') as f:
            pickle.dump(entry, f)

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

        path = os.path.join(self.save_dir, filename)
        images = make_grid_with_labels(images, task_structures, nrow=n_per_batch, normalize=True, limit=None,
                                       channels_first=True)
        logger.info(f"Saving layout thumbnails to {path}.")
        vutils.save_image(images, path)

    def check_unique(self, batches):

        all_data = []
        for batch in batches:
            all_data.extend(batch.features)
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
            instance = super().__new__(WaveCollapseBatch)
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

        batch_data = self.generate_data()
        if isinstance(batch_data, dgl.DGLGraph) or isinstance(batch_data[0], dgl.DGLGraph):
            features, _ = util.get_node_features(batch_data, device=None) # TODO: add device?
        else:
            features = batch_data

        if not util.check_unique(features).all():
            logger.warning(f"Batch {self.batch_meta['batch_id']} generated duplicate features.")
            self.batch_meta['unique_data'] = False
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
            self.label_contents["images"] = tr.Nav2DTransforms.dense_graph_to_minigrid_render(data, tile_size=16, level_info=self.dataset_meta.level_info)

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

        start_dim, goal_dim = self.dataset_meta.config.graph_feature_descriptors.index('start'), \
                              self.dataset_meta.config.graph_feature_descriptors.index('goal')

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
            features = tr.Nav2DTransforms.minigrid_layout_to_dense_graph(features, to_dgl=False, remove_edges=False)

            if self.dataset_meta.ensure_connected:
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
            layouts = np.ones(patterns.shape, dtype=self.dataset_meta.level_info['dtype']) * tr.Minigrid_OBJECT_TO_IDX['empty']

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
                for j in range(len(self.dataset_meta.config.graph_feature_descriptors)):
                    nx.set_node_attributes(g, wall_graph_attr[j], self.dataset_meta.config.graph_feature_descriptors[j])
                nx.set_node_attributes(g, act, "active")
                g.add_edges_from(component.edges(data=True))
                if to_dgl:
                    g = dgl.from_networkx(g, node_attrs=self.dataset_meta.config.graph_feature_descriptors)
                graphs[i] = copy.deepcopy(g)

            return graphs

        def get_images(self, idx: List[int]) -> List[torch.Tensor]:

            data = [self.features[i] for i in idx]
            images = tr.Nav2DTransforms.dense_graph_to_minigrid_render(data, tile_size=16, level_info=self.dataset_meta.level_info)
            return images


if __name__ == '__main__':
    generate_dataset()