import logging

import dgl
import einops
import networkx as nx
import numpy as np
from mazelib import Maze
from typing import Union, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from data_generators import OBJECT_TO_CHANNEL_AND_IDX, OBJECT_TO_FEATURE_DIM
from gym_minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX as Minigrid_OBJECT_TO_IDX, \
    IDX_TO_OBJECT as Minigrid_IDX_TO_OBJECT

logger = logging.getLogger(__name__)

class BinaryTransform(object):
    def __init__(self, thr):
        self.thr = thr

    def __call__(self, x):
        return (x >= self.thr).to(x)  # do not change the data type or device


class FlipBinaryTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.abs(x - 1)


class ReshapeTransform(object):
    def __init__(self, *args):
        self.shape = args

    def __call__(self, x: torch.Tensor):
        return x.view(*self.shape)


class FlattenTransform(object):
    def __init__(self, *args):
        self.dims = args  # start and end dim

    def __call__(self, x: torch.Tensor):
        return x.view(*self.dims)


class DilationTransform(object):
    def __init__(self, num_dilations: int = 1):
        self.num_dilations = num_dilations

    def __call__(self, x: torch.Tensor):
        # x has shape (B, C, H, W)
        n_channels = x.shape[1]
        weight = torch.ones((n_channels, 1, 1, 1))
        stride = 1 + self.num_dilations
        pad = [self.num_dilations for i in range(4)]

        out = F.conv_transpose2d(x, weight=weight, stride=stride, groups=n_channels)
        out = F.pad(out, pad)
        return out


class SelectChannelsTransform(object):
    def __init__(self, *args):
        self.selected_channels = args

    def __call__(self, x: torch.Tensor):
        return x[..., self.selected_channels]


class ToDeviceTransform(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, x: torch.Tensor):
        return x.to(self.device)


class Nav2DTransforms:
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
            maze.grid = np.int8(grid[..., 0])
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
        grid_layout_dim = (
        gridworlds.shape[0], int(np.floor(gridworlds.shape[1] / 2)), int(np.floor(gridworlds.shape[2] / 2)), 2)
        grid_layouts = np.zeros(grid_layout_dim)

        layout_channel = OBJECT_TO_CHANNEL_AND_IDX['empty'][0]
        empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]

        for m in range(grid_layouts.shape[0]):
            for i in range(grid_layouts.shape[1]):
                for j in range(grid_layouts.shape[2]):
                    ind_gridworld = (m, int(i * 2 + 1), int(j * 2 + 1), layout_channel)
                    ind_gridworld_right = list(ind_gridworld)
                    ind_gridworld_right[2] += 1
                    ind_gridworld_bot = list(ind_gridworld)
                    ind_gridworld_bot[1] += 1
                    if gridworlds[ind_gridworld] == empty_idx:
                        if gridworlds[tuple(ind_gridworld_right)] == empty_idx:
                            grid_layouts[m, i, j, 0] = 1
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
        grids = np.stack((grid_layouts[..., 0], grid_layouts[..., 1], start_channels, goal_channels), axis=-1)
        return grids

    @staticmethod
    def encode_grid_to_gridworld(grids: Union[np.ndarray, torch.Tensor], layout_only=False):

        if layout_only:
            expected_channels = 2
        else:
            expected_channels = 4

        # TODO: perf remodel to handle GPU
        tensor = False
        if torch.is_tensor(grids):
            tensor = True
            device = grids.device
            assert len(grids.shape) == 4, f"Grids Tensor has {len(grids.shape)} dimensions. Expected {4}"
            if grids.shape[-1] != expected_channels and grids.shape[1] == expected_channels:
                grids = torch.permute(grids, (0, 2, 3, 1))  # (B, C, H, W) -> (B, H, W, C)
            grids = grids.detach().cpu().numpy()

        assert grids.shape[
                   -1] == expected_channels, f"Inputted Grids have {grids.shape[-1]} channels. Expected {expected_channels}"

        gridworlds_layout_dim = (
            grids.shape[0], int(2 * grids.shape[1] + 1), int(2 * grids.shape[2] + 1))
        gridworlds_layouts = np.ones(gridworlds_layout_dim) * OBJECT_TO_CHANNEL_AND_IDX['wall'][-1]

        gridworlds_layout_channel = OBJECT_TO_CHANNEL_AND_IDX['empty'][0]  # could be used with a "sort"
        gridworlds_empty_idx = OBJECT_TO_CHANNEL_AND_IDX['empty'][-1]

        for m in range(grids.shape[0]):
            for i in range(grids.shape[1]):
                for j in range(grids.shape[2]):
                    if grids[m, i, j, 0] == 1 or grids[m, i, j, 1] == 1:
                        i_gridworld, j_gridworld = (2 * i + 1, 2 * j + 1)
                        gridworlds_layouts[m, i_gridworld, j_gridworld] = gridworlds_empty_idx
                        if grids[m, i, j, 0] == 1:
                            gridworlds_layouts[m, i_gridworld, j_gridworld + 1] = gridworlds_empty_idx
                            gridworlds_layouts[m, i_gridworld, j_gridworld + 2] = gridworlds_empty_idx
                        if grids[m, i, j, 1] == 1:
                            gridworlds_layouts[m, i_gridworld + 1, j_gridworld] = gridworlds_empty_idx
                            gridworlds_layouts[m, i_gridworld + 2, j_gridworld] = gridworlds_empty_idx
                        # clique rule
                        if grids[m, i, j, 0] == grids[m, i, j, 1] == grids[m, i + 1, j, 0] == grids[
                            m, i, j + 1, 1] == 1:
                            gridworlds_layouts[m, i_gridworld + 1, j_gridworld + 1] = gridworlds_empty_idx

        if layout_only:
            gridworlds = np.reshape(gridworlds_layouts, (*gridworlds_layouts.shape, 1))
        # DROPED: use object dictionary
        else:
            start_channels, goal_channels = (np.zeros(gridworlds_layout_dim) for i in range(2))

            start_inds_grids = np.where(grids[..., 2] == 1)
            start_inds_gridworlds = (start_inds_grids[0], (2 * start_inds_grids[1] + 1).astype(int),
                                     (2 * start_inds_grids[2] + 1).astype(int))
            goal_inds_grids = np.where(grids[..., 3] == 1)
            goal_inds_gridworlds = (goal_inds_grids[0], (2 * goal_inds_grids[1] + 1).astype(int),
                                    (2 * goal_inds_grids[2] + 1).astype(int))

            start_channels[start_inds_gridworlds] = OBJECT_TO_CHANNEL_AND_IDX['start'][1]
            goal_channels[goal_inds_gridworlds] = OBJECT_TO_CHANNEL_AND_IDX['goal'][1]

            # merge
            gridworlds = np.stack((gridworlds_layouts, start_channels, goal_channels), axis=-1)

        if tensor:
            gridworlds = torch.tensor(gridworlds, dtype=torch.float, device=device)
            gridworlds = torch.permute(gridworlds, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)

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

        adj = Nav2DTransforms.encode_gridworld_layout_to_adj(gridworlds[..., layout_channel], empty_idx)  # M, N, N

        start_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['start'][0]]
                                        == OBJECT_TO_CHANNEL_AND_IDX['start'][1])
        goal_inds_gridworld = np.where(gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['goal'][0]]
                                       == OBJECT_TO_CHANNEL_AND_IDX['goal'][1])

        start_nodes_graph = (start_inds_gridworld[0],
                             ((start_inds_gridworld[1] - 1) / 2 * dim_grid[0] + (
                                         start_inds_gridworld[2] - 1) / 2).astype(int))
        goal_nodes_graph = (goal_inds_gridworld[0],
                            ((goal_inds_gridworld[1] - 1) / 2 * dim_grid[0] + (goal_inds_gridworld[2] - 1) / 2).astype(
                                int))
        active_nodes_graph = np.where(adj.sum(axis=1) != 0)
        wall_nodes_graph = np.where(adj.sum(axis=1) == 0)

        feats = np.zeros((adj.shape[0], adj.shape[1], 4))  # M, N, D
        feats[(*start_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['start']] * len(start_nodes_graph[0])))] = 1
        feats[(*goal_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['goal']] * len(goal_nodes_graph[0])))] = 1
        if wall_nodes_graph[0].size != 0:  # only if array not empty.
            feats[(*wall_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['wall']] * len(wall_nodes_graph[0])))] = 1
        # empty features are the active nodes, removing the nodes having goal or start feature
        feats[(*active_nodes_graph, np.array([OBJECT_TO_FEATURE_DIM['empty']] * len(active_nodes_graph[0])))] = 1
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

    @staticmethod
    def encode_decoder_output_to_graph(logits_A: torch.Tensor, logits_Fx: torch.Tensor, decoder,
                                     correct_A: bool = False):

        mode_A, mode_Fx = decoder.param_m((logits_A, logits_Fx))

        logger.info(f"encode_decoder_output_to_graph(): Executed mode_A, mode_Fx = decoder.param_m((logits_A, logits_Fx)).")

        start_dim = decoder.attributes.index('start')
        goal_dim = decoder.attributes.index('goal')
        n_nodes = mode_Fx.shape[1]

        # TODO correct_Fx
        mode_A = mode_A.reshape(mode_A.shape[0], -1, 2)

        logger.info(f"encode_decoder_output_to_graph(): Executed mode_A = mode_A.reshape(mode_A.shape[0], -1, 2).")

        start_nodes = mode_Fx[..., start_dim].argmax(dim=-1)
        goal_nodes = mode_Fx[..., goal_dim].argmax(dim=-1)

        logger.info(f"encode_decoder_output_to_graph(): Executed goal_nodes = mode_Fx[..., goal_dim].argmax(dim=-1).")
        is_valid, adj = Nav2DTransforms._check_validity(mode_A, start_nodes, goal_nodes, n_nodes, correct_A=correct_A)
        logger.info(f"encode_decoder_output_to_graph(): Executed Nav2DTransforms.check_validity().")
        #DEBUG: CRASHING HERE *2
        mode_Fx = mode_Fx.cpu()
        adj = adj.cpu().numpy()

        logger.info(f"encode_decoder_output_to_graph(): Executed mode_Fx = mode_Fx.cpu().")
        graphs = []
        for m in range(adj.shape[0]):
            src, dst = np.nonzero(adj[m])
            g = dgl.graph((src, dst), num_nodes=len(mode_Fx[m]))
            g.ndata['feat'] = mode_Fx[m]
            graphs.append(g)
        logger.info(f"encode_decoder_output_to_graph(): Executed graphs.append(g) m times.")

        return graphs, start_nodes, goal_nodes, is_valid

    @staticmethod
    def encode_graph_to_gridworld(graphs: Union[dgl.DGLGraph, List[dgl.DGLGraph], tuple],
                                  attributes: Tuple[str] = ("empty", "wall", "start", "goal"),
                                  used_attributes: Tuple[str] = ("start", "goal"),
                                  probabilistic_mode: bool = False,
                                  output_dtype: str = 'tensor'):
        # TODO: probabilistic_mode not fully implemented
        # TODO: perf enhancements if going full tensor

        def get_gw_inds(nodes_tuple: Tuple[np.ndarray], n_nodes, mapping=lambda x: 2 * x + 1):
            inds_tuple = []
            for nodes in nodes_tuple:
                inds = (nodes[1] // np.sqrt(n_nodes), nodes[1] % np.sqrt(n_nodes))
                inds = tuple([mapping(i.astype(int)) for i in inds])
                inds = (nodes[0],) + inds
                inds_tuple.append(inds)
            return tuple(inds_tuple)

        # Note: modes 2 and 3 can only work for layouts with 1-to-1 cell-node characterisation
        possible_modes = {
            ()                                : 0, ("",): 0,  # 0: Layout only from A
            ("start", "goal")                 : 1,  # 1: Layout from A, start and goal from Fx
            ("empty", "start", "goal")        : 2,  # 2: Layout, start, goal from Fx
            ("empty", "wall", "start", "goal"): 3,  # 3: Layout, start, goal from Fx, may form impossible layouts
            }

        try:
            mode = possible_modes[tuple(used_attributes)]
        except KeyError:
            raise AttributeError(f"Gridworld encoding from {used_attributes} is not possible.")

        if isinstance(graphs, tuple):
            A, Fx = graphs
            # n_nodes = Fx.shape[-2]
            n_nodes = int(A.shape[
                              -1] / 2 + 1)  # TODO: find a better way to handle full graph encoding (may require an additional input argument)
            A = A.reshape(A.shape[0], -1, 2)
            A = A.cpu().numpy()  # TODO make more efficient to handle tensors
            if Fx is not None:
                Fx = Fx.cpu().numpy()
        elif isinstance(graphs, dgl.DGLGraph) or isinstance(graphs[0], dgl.DGLGraph):
            if isinstance(graphs, dgl.DGLGraph): graphs = dgl.unbatch(graphs)
            n_nodes = graphs[0].num_nodes()  # assumption that all graphs have same number of nodes
            feat_dim = graphs[0].ndata['feat'].shape
            assert n_nodes % np.sqrt(n_nodes) == 0  # we are assuming square layout

            A = np.empty((len(graphs), n_nodes, n_nodes))
            Fx = np.empty((len(graphs), *feat_dim))
            for m in range(len(graphs)):
                A[m] = graphs[m].adj().cpu().to_dense().numpy()
                Fx[m] = graphs[m].ndata['feat'].cpu().numpy()
            A = Nav2DTransforms.encode_adj_to_reduced_adj(A)
        else:
            raise RuntimeError(f"data format {type(graphs)} is not supported by function. Format supported are"
                               f"List[dgl.DGLGraph], tuple[tensor, tensor]")

        if output_dtype == 'tensor':
            device = graphs[0].device

        gridworld_layout_dim = (int(2 * np.sqrt(n_nodes) + 1), int(2 * np.sqrt(n_nodes) + 1))

        # Modes for which we need A
        if mode in [0, 1]:
            gridworlds_layouts = Nav2DTransforms.encode_reduced_adj_to_gridworld_layout(A, gridworld_layout_dim,
                                                                                        probalistic_mode=probabilistic_mode)
            if mode in [0, ]:
                gridworlds = np.reshape(gridworlds_layouts, (*gridworlds_layouts.shape, 1))
        # Modes for which we need Fx[start, goal]
        if mode in [1, 2, ]:  # [1,2,3] when implemented
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
            if mode in [1, ]:  # add layout from adjacency
                gridworlds[..., OBJECT_TO_CHANNEL_AND_IDX['wall'][0]] = OBJECT_TO_CHANNEL_AND_IDX['wall'][1] * \
                                                                        gridworlds_layouts
            elif mode in [2, ]:  # add layout from empty nodes
                # set all cells to wall in layout channel
                gridworlds[..., np.array([OBJECT_TO_CHANNEL_AND_IDX['wall'][0]])] = OBJECT_TO_CHANNEL_AND_IDX['wall'][1]

                # set all non wall (empty, goal, start) to empty
                # start, goal
                gridworlds[(*start_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * gridworlds.shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]
                gridworlds[(*goal_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * gridworlds.shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]

                # empty
                empty_nodes = np.where(Fx[..., attributes.index('empty')] == 1)
                # empty_inds = (empty_nodes[1] // np.sqrt(n_nodes), empty_nodes[1] % np.sqrt(n_nodes))
                # empty_inds = tuple([2 * i.astype(int) + 1 for i in empty_inds])
                # empty_inds = (empty_nodes[0],) + goal_inds
                empty_inds, = get_gw_inds((empty_nodes,), n_nodes)

                gridworlds[(*empty_inds, np.array([OBJECT_TO_CHANNEL_AND_IDX['empty'][0]] * empty_inds[0].shape[0]))] = \
                    OBJECT_TO_CHANNEL_AND_IDX['empty'][1]
        elif mode in [3, ]:
            raise NotImplementedError(f"Gridworld encoding from {used_attributes} not yet implemented.")

        if output_dtype == 'tensor':
            gridworlds = torch.tensor(gridworlds, dtype=torch.float, device=device)
            gridworlds = torch.permute(gridworlds, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)

        return gridworlds

    @staticmethod
    def encode_reduced_adj_to_gridworld_layout(A: Union[np.ndarray, torch.tensor], layout_dim, probalistic_mode=False,
                                               prob_threshold=0.5):

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
                cell_val = max(A[m, n, :].max(), gridworlds_layouts[
                    m, i, j])  # node likelihood is max of (max edge probability and previous inputed max edge probability)
                gridworlds_layouts[m, i, j] = cell_val
                # horizontal edges
                if (j + 2) < gridworlds_layouts.shape[2]:
                    edge_val = A[m, n, 0]
                    cell_val = max(A[m, n, 0], gridworlds_layouts[
                        m, i, j + 2])  # update node likelihood if an edge of higher probability is found
                    gridworlds_layouts[m, i, j + 1] = edge_val
                    gridworlds_layouts[m, i, j + 2] = cell_val
                # vertical edges
                if (i + 2) < gridworlds_layouts.shape[1]:
                    edge_val = A[m, n, 1]
                    cell_val = max(A[m, n, 1], gridworlds_layouts[m, i + 2, j])
                    gridworlds_layouts[m, i + 1, j] = edge_val
                    gridworlds_layouts[m, i + 2, j] = cell_val
                # clique rule
                if i + 1 < gridworlds_layouts.shape[1] and j + 1 < gridworlds_layouts.shape[2]:
                    if n + int(np.sqrt(n_nodes)) < A[m].shape[0]:
                        clique = torch.tensor(
                            [A[m, n, 0], A[m, n, 1], A[m, n + 1, 1], A[m, n + int(np.sqrt(n_nodes)), 0]])
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
        # layouts shape: [m, odd, odd]
        assert layouts.shape[1] % 2 == 1 and layouts.shape[2] % 2 == 1, \
            "Inputted Gridworlds Layouts do not have odd number of elements"
        assert len(layouts.shape) == 3, "Layout not inputted correctly. Input layouts as (m, row, col)"
        node_inds_i, node_inds_j = [i for i in range(1, layouts.shape[1], 2)], [i for i in
                                                                                range(1, layouts.shape[2], 2)]
        A = np.zeros((layouts.shape[0], len(node_inds_i) * len(node_inds_j), len(node_inds_i) * len(node_inds_j)))

        for m in range(A.shape[0]):
            for i_A, i_gw in enumerate(node_inds_i):
                for j_A, j_gw in enumerate(node_inds_j):
                    if layouts[m, i_gw, j_gw] == empty_idx:
                        ind_gw_right = (m, i_gw, j_gw + 1)
                        ind_gw_bot = (m, i_gw + 1, j_gw)
                        if layouts[ind_gw_right] == empty_idx:
                            ind_A_right = (m, i_A * len(node_inds_i) + j_A, i_A * len(node_inds_i) + j_A + 1)
                            A[ind_A_right] = 1
                        if layouts[ind_gw_bot] == empty_idx:
                            ind_A_bot = (m, i_A * len(node_inds_i) + j_A, (i_A + 1) * len(node_inds_i) + j_A)
                            A[ind_A_bot] = 1
            A[m] = np.triu(A[m]) + np.tril(A[m].T, 1)

        return A

    @staticmethod
    def encode_adj_to_reduced_adj(adj: Union[np.ndarray, torch.tensor]):
        # the last sqrt(n)-1 col edges will always be 0.
        # adj shape (m, n, n)
        if torch.is_tensor(adj):
            A = torch.zeros((adj.shape[0], adj.shape[1] - 1, 2)).to(adj.device)
        elif isinstance(adj, np.ndarray):
            A = np.zeros((adj.shape[0], adj.shape[1] - 1, 2))
        dim_grid = int(np.sqrt(adj.shape[1]))  # only for square grids
        A[..., 0] = adj.diagonal(1, 1, 2)  # row edges
        A[:, :-dim_grid + 1, 1] = adj.diagonal(dim_grid, 1, 2)  # col edges

        return A  # (m, n-1, 2)

    @staticmethod
    def encode_reduced_adj_to_adj(adj_r: np.ndarray):
        #DEBUG: CRASHING
        # only for square grids
        # adj shape (m, n-1, 2)
        A = np.empty((adj_r.shape[0], adj_r.shape[1] + 1, adj_r.shape[1] + 1))
        dim_grid = int(np.sqrt(A.shape[1]))
        for m in range(A.shape[0]):
            A[m] = np.diag(adj_r[m, :, 0], k=1)
            A[m] += np.diag(adj_r[m, :-dim_grid + 1, 1], k=dim_grid)
            A[m] = np.triu(A[m]) + np.tril(A[m].T, 1)

        return A  # (m, n, n)

    @staticmethod
    def augment_adj(n: int, transforms: torch.tensor):
        # transforms represent all allowable permutations in a 2D grid
        nodes_inds = torch.arange(0, n, dtype=torch.int)  # node indices in adjacency matrix
        i_n, j_n = nodes_inds.div(int(n ** .5), rounding_mode='floor'), nodes_inds % int(
            n ** .5)  # corresponding indices in grid space, top left corner origin
        ij_n = torch.stack([i_n, j_n], dim=0)  # D N
        ij_n = einops.repeat(ij_n, 'd n ->  p d n', p=transforms.shape[0])  # P D N
        c = torch.tensor([int((n ** .5 - 1) / 2), int((n ** .5 - 1) / 2)], dtype=torch.int).unsqueeze(1)  # D 1
        c = einops.repeat(c, 'd n -> p d n', p=transforms.shape[0])  # P D 1
        ij_c = (ij_n - c)  # P D=2 N. Corresponding indices with origin in the middle of the grid
        # coordinate transform 1 done: origin in grid space
        ij_t = torch.matmul(transforms, ij_c)  # P 2 2 @ P D=2 N -> P D=2 N # rotate the coordinate axis
        # coordinate transform 2 done: axis rotated
        ij_f = (ij_t + c)  # add the centroid to come back to a top left corner coordinate system
        ij2n = torch.tensor([int(n ** 0.5), 1], dtype=torch.int).unsqueeze(
            1)  # D 1 #transformation matrix to get node ordering, left to right, top to bottom in graph space
        ij2n = einops.repeat(ij2n, 'd n -> p n d', p=transforms.shape[0])  # P 1 D
        nodes_inds_t = torch.matmul(ij2n, ij_f).squeeze()  # P 1 D @ P D N = P D N #recover the transformed indices
        return nodes_inds_t

    @staticmethod
    def check_validity_start_goal(start_nodes: torch.Tensor, goal_nodes: torch.Tensor, A:torch.Tensor, threshold:int=0.5) -> torch.Tensor:
        """
        Check if the start and goal nodes are valid.
        """
        logger.info("check_validity_start_goal(): Entering function")
        logger.info(f"check_validity_start_goal(): Input shapes: start_nodes {start_nodes.shape}, goal_nodes {goal_nodes.shape}, A {A.shape}")
        # device = start_nodes.device
        # start_nodes = start_nodes.cpu()
        # goal_nodes = goal_nodes.cpu()
        # A = A.cpu()
        #
        # logger.info("check_validity_start_goal(): Moved, start_nodes, goal_nodes, A to cpu")

        batch_inds = torch.arange(0, start_nodes.shape[0])

        mask1 = start_nodes == goal_nodes
        mask2 = A[batch_inds, start_nodes].amax(dim=-1) < threshold
        mask3 = A[batch_inds, goal_nodes].amax(dim=-1) < threshold
        # valid only if NOT(start==goal OR no edges from start OR no edges from goal)
        valid = ~(mask1 | mask2 | mask3)

        return valid
        # return valid.to(device)

    @staticmethod
    def check_invalid_edges_reduced_A(A, n_nodes, correct_A=False, threshold=0.5):
        """
        Checks for and optionally corrects invalid edges in the reduced adjacency matrix. Will always return true if
        correct_A flag is true. Does not check if there are no edges between nodes.
        """

        if len(A.shape) == 2:
            A = A.reshape(A.shape[0], -1, 2)

        assert len(A.shape) == 3
        assert A.shape[1], A.shape[2] == (n_nodes - 1, 2)

        n_root = np.sqrt(n_nodes)
        seq = np.arange(1, n_root)
        inds_grid_right = seq * n_root - 1
        # Note: the -1 below achieves 2 things:
        # 1) catch the first element of the last row,
        # 2) account for the last node (bottom right corner) not being present in A reduced
        inds_grid_bot = (n_root - 1) * n_root + seq - 1

        if (A[:, inds_grid_bot, 1] >= threshold).any() or (A[:, inds_grid_right, 0] >= threshold).any():
            if correct_A:
                A[:, inds_grid_bot, 1] = 0
                A[:, inds_grid_right, 0] = 0
                valid = torch.tensor([True] * A.shape[0], dtype=torch.bool, device=A.device)
            else:
                mask = (A[:, inds_grid_bot, 1] >= threshold).sum(axis=-1) + (A[:, inds_grid_right, 0] >= threshold).sum(
                    axis=-1)
                valid = ~mask.to(torch.bool)
        else:
            valid = torch.tensor([True] * A.shape[0], dtype=torch.bool, device=A.device)

        return A, valid

    @staticmethod
    def _check_validity(A_red, start_nodes, goal_nodes, n_nodes, correct_A=False, threshold=0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if the reduced adjacency matrix is valid and checks if the start and goal nodes are placed correctly

        Returns:
            - valid: an boolean tensor of shape (batch_size,) indicating whether the reduced adjacency matrix is valid
            - A: the full adjacency matrices. Corrected if correct_A is True.
        """
        logger.debug(f"check_validity(): DOES DEBUG WORK HERE??")
        logger.info(f"check_validity(): Entered function.")
        logger.info(f"check_validity(): Input shapes:")
        logger.info(f"A_red:{A_red.shape}")
        logger.info(f"start_nodes{start_nodes.shape}")
        logger.info(f"goal_nodes{goal_nodes.shape}")
        logger.info(f"n_nodes={n_nodes}")
        logger.info(f"correct_A={correct_A}")
        logger.info(f"threshold={threshold}")
        A_red, valid_A = Nav2DTransforms.check_invalid_edges_reduced_A(A_red, n_nodes, correct_A, threshold=threshold)
        logger.info(f"check_validity(): Executed check_invalid_edges_reduced_A().")
        # DEBUG: crash here
        A = Nav2DTransforms.encode_reduced_adj_to_adj(A_red.cpu().numpy())
        A = torch.tensor(A, device=start_nodes.device, dtype=torch.float)
        valid_start_goal = Nav2DTransforms.check_validity_start_goal(start_nodes, goal_nodes, A, threshold=threshold)
        logger.info(f"check_validity(): Executed check_validity_start_goal().")
        valid = valid_A & valid_start_goal

        return valid, A

    @staticmethod
    def force_valid_layout(graphs: List[nx.Graph], logits_start, logits_goal: torch.tensor):
        """
        Given a fully connected graph (previously reduced to its principal component), ensures the generated layouts
        valid by forcing the start and goal node to be on the largest connected component.
        (but not at the same position).
        """

        all_nodes = set(range(logits_goal.shape[-1]))
        is_valid = [True] * len(logits_start)
        for m, graph in enumerate(graphs):
            connected_nodes = set(graph.nodes)
            to_remove_nodes_start = list(all_nodes - connected_nodes)
            start_node = logits_start[m].argmax()
            to_remove_nodes_goal = list(all_nodes - connected_nodes - {start_node.item()})

            if to_remove_nodes_goal==all_nodes or to_remove_nodes_start==all_nodes:
                is_valid[m] = False
                continue

            logits_start[m, to_remove_nodes_start] = 0.
            logits_goal[m, to_remove_nodes_goal] = 0.

        start_nodes = logits_start.argmax(dim=-1)
        goal_nodes = logits_goal.argmax(dim=-1)
        start_onehot = F.one_hot(start_nodes, num_classes=logits_start.shape[-1]).to(logits_start)
        goal_onehot = F.one_hot(goal_nodes, num_classes=logits_goal.shape[-1]).to(logits_goal)

        return start_onehot, goal_onehot, start_nodes.tolist(), goal_nodes.tolist(), is_valid

