from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import product

import networkx as nx
import numpy as np

from minigrid.core.constants import COLOR_TO_IDX, IDX_TO_OBJECT, OBJECT_TO_IDX
from minigrid.minigrid_env import MiniGridEnv


@dataclass
class EdgeDescriptor:
    between: tuple[str, str] | tuple[str]
    structure: str | None = None


# This is maybe general enough to be in utils
class GraphTransforms:
    OBJECT_TO_DENSE_GRAPH_ATTRIBUTE = {
        "empty": ("navigable", "empty"),
        "start": ("navigable", "start"),
        "agent": ("navigable", "start"),
        "goal": ("navigable", "goal"),
        "moss": ("navigable", "moss"),
        "wall": ("non_navigable", "wall"),
        "lava": ("non_navigable", "lava"),
    }

    DENSE_GRAPH_ATTRIBUTE_TO_OBJECT = {
        "empty": "empty",
        "start": "start",
        "goal": "goal",
        "moss": "moss",
        "wall": "wall",
        "lava": "lava",
        "navigable": None,
        "non_navigable": None,
    }

    MINIGRID_COLOR_CONFIG = {
        "empty": None,
        "wall": "grey",
        "agent": "blue",
        "goal": "green",
        "lava": "red",
        "moss": "purple",
    }

    @staticmethod
    def minigrid_to_bitmap(grids):
        layout = grids[..., 0]
        bitmap = np.zeros_like(layout)
        bitmap[layout == 2] = 1
        bitmap = list(bitmap)

        start_pos_id = np.where(layout == 10)
        goal_pos_id = np.where(layout == 8)

        start_pos = []
        goal_pos = []
        for i in range(len(bitmap)):
            bitmap[i] = bitmap[i][1:-1, 1:-1]
            start_pos.append(np.array([start_pos_id[2][i], start_pos_id[1][i]]))
            goal_pos.append(np.array([goal_pos_id[2][i], goal_pos_id[1][i]]))

        return bitmap, start_pos, goal_pos

    @staticmethod
    def minigrid_to_dense_graph(
        minigrids: np.ndarray | list[MiniGridEnv],
        node_attr=None,
        edge_config=None,
    ) -> list[nx.Graph]:
        if isinstance(minigrids[0], np.ndarray):
            minigrids = np.array(minigrids)
            layouts = minigrids[..., 0]
        elif isinstance(minigrids[0], MiniGridEnv):
            layouts = [minigrid.grid.encode()[..., 0] for minigrid in minigrids]
            for i in range(len(minigrids)):
                layouts[i][tuple(minigrids[i].agent_pos)] = OBJECT_TO_IDX["agent"]
            layouts = np.array(layouts)
        else:
            raise TypeError(
                f"minigrids must be of type List[np.ndarray], List[MiniGridEnv], "
                f"List[MultiGridEnv], not {type(minigrids[0])}"
            )
        graphs, _ = GraphTransforms.minigrid_layout_to_dense_graph(
            layouts, remove_border=True, node_attr=node_attr, edge_config=edge_config
        )
        return graphs

    @staticmethod
    def minigrid_layout_to_dense_graph(
        layouts: np.ndarray, remove_border=True, node_attr=None, edge_config=None
    ) -> tuple[list[nx.Graph], dict[str, list[nx.Graph]]]:
        assert (
            layouts.ndim == 3
        ), f"Wrong dimensions for minigrid layout, expected 3 dimensions, got {layouts.ndim}."

        node_attr = [] if node_attr is None else node_attr

        # Remove borders
        if remove_border:
            layouts = layouts[:, 1:-1, 1:-1]  # remove edges
        dim_grid = layouts.shape[1:]

        # Get the objects present in the layout
        objects_idx = np.unique(layouts)
        object_instances = [IDX_TO_OBJECT[obj] for obj in objects_idx]
        assert set(object_instances).issubset(
            {"empty", "wall", "start", "goal", "agent", "lava", "moss"}
        ), (
            f"Unsupported object(s) in minigrid layout. Supported objects are: "
            f"empty, wall, start, goal, agent, lava, moss. Got {object_instances}."
        )

        # Get location of each object in the layout
        object_locations = {}
        for obj in object_instances:
            object_locations[obj] = defaultdict(list)
            ids = list(zip(*np.where(layouts == OBJECT_TO_IDX[obj])))
            for tup in ids:
                object_locations[obj][tup[0]].append(tup[1:])
            for m in range(layouts.shape[0]):
                if m not in object_locations[obj]:
                    object_locations[obj][m] = []
            object_locations[obj] = OrderedDict(sorted(object_locations[obj].items()))
        if "start" not in object_instances and "agent" in object_instances:
            object_locations["start"] = object_locations["agent"]
        if "agent" not in object_instances and "start" in object_instances:
            object_locations["agent"] = object_locations["start"]

        # Create one-hot graph feature tensor
        graph_feats = {}
        object_to_attr = GraphTransforms.OBJECT_TO_DENSE_GRAPH_ATTRIBUTE
        for obj in object_instances:
            for attr in object_to_attr[obj]:
                if attr not in graph_feats and attr in node_attr:
                    graph_feats[attr] = np.zeros(layouts.shape)
                loc = list(object_locations[obj].values())
                assert len(loc) == layouts.shape[0]
                for m in range(layouts.shape[0]):
                    if loc[m]:
                        loc_m = np.array(loc[m])
                        graph_feats[attr][m][loc_m[:, 0], loc_m[:, 1]] = 1
        for attr in node_attr:
            if attr not in graph_feats:
                graph_feats[attr] = np.zeros(layouts.shape)
            graph_feats[attr] = graph_feats[attr].reshape(layouts.shape[0], -1)

        graphs, edge_graphs = GraphTransforms.features_to_dense_graph(
            graph_feats, dim_grid, edge_config
        )

        return graphs, edge_graphs

    @staticmethod
    def features_to_dense_graph(
        features: dict[str, np.ndarray],
        dim_grid: tuple,
        edge_config: dict[str, EdgeDescriptor] = None,
    ) -> tuple[list[nx.Graph], dict[str, list[nx.Graph]]]:
        graphs = []
        edge_graphs = defaultdict(list)
        for m in range(features[list(features.keys())[0]].shape[0]):
            g_temp = nx.grid_2d_graph(*dim_grid)
            g = nx.Graph()
            g.add_nodes_from(sorted(g_temp.nodes(data=True)))
            for attr in features:
                nx.set_node_attributes(
                    g, {k: v for k, v in zip(g.nodes, features[attr][m].tolist())}, attr
                )
            if edge_config is not None:
                edge_layers = GraphTransforms.get_edge_layers(
                    g, edge_config, list(features.keys()), dim_grid
                )
                for edge_n, edge_g in edge_layers.items():
                    g.add_edges_from(edge_g.edges(data=True), label=edge_n)
                    edge_graphs[edge_n].append(edge_g)
            graphs.append(g)

        return graphs, edge_graphs

    @staticmethod
    def graph_features_to_minigrid(
        graph_features: dict[str, np.ndarray], shape: tuple[int, int], padding=1
    ) -> np.ndarray:
        features = graph_features.copy()
        node_attributes = list(features.keys())

        color_config = GraphTransforms.MINIGRID_COLOR_CONFIG

        # shape_no_padding = (features[node_attributes[0]].shape[-2], shape[0] - 2, shape[1] - 2, 3)
        shape_no_padding = (shape[0] - 2 * padding, shape[1] - 2 * padding, 3)
        for attr in node_attributes:
            features[attr] = features[attr].reshape(*shape_no_padding[:-1])
        grids = np.ones(shape_no_padding, dtype=np.uint8) * OBJECT_TO_IDX["empty"]

        minigrid_object_to_encoding_map = {}  # [object_id, color, state]
        for feature in node_attributes:
            obj_type = GraphTransforms.DENSE_GRAPH_ATTRIBUTE_TO_OBJECT[feature]
            if (
                obj_type is not None
                and obj_type not in minigrid_object_to_encoding_map.keys()
            ):
                if obj_type == "empty":
                    minigrid_object_to_encoding_map[obj_type] = [
                        OBJECT_TO_IDX["empty"],
                        0,
                        0,
                    ]
                elif obj_type == "agent":
                    minigrid_object_to_encoding_map[obj_type] = [
                        OBJECT_TO_IDX["agent"],
                        0,
                        0,
                    ]
                elif obj_type == "start":
                    color_str = color_config["agent"]
                    minigrid_object_to_encoding_map[obj_type] = [
                        OBJECT_TO_IDX["agent"],
                        COLOR_TO_IDX[color_str],
                        0,
                    ]
                else:
                    color_str = color_config[obj_type]
                    minigrid_object_to_encoding_map[obj_type] = [
                        OBJECT_TO_IDX[obj_type],
                        COLOR_TO_IDX[color_str],
                        0,
                    ]

        if (
            "start" not in minigrid_object_to_encoding_map.keys()
            and "agent" in minigrid_object_to_encoding_map.keys()
        ):
            minigrid_object_to_encoding_map["start"] = minigrid_object_to_encoding_map[
                "agent"
            ]
        if (
            "agent" not in minigrid_object_to_encoding_map.keys()
            and "start" in minigrid_object_to_encoding_map.keys()
        ):
            minigrid_object_to_encoding_map["agent"] = minigrid_object_to_encoding_map[
                "start"
            ]

        for i, attr in enumerate(node_attributes):
            if "wall" not in node_attributes:
                if attr == "navigable" and "wall" not in node_attributes:
                    mapping = minigrid_object_to_encoding_map["wall"]
                    grids[features[attr] == 0] = np.array(mapping, dtype=np.uint8)
                else:
                    mapping = minigrid_object_to_encoding_map[attr]
                    grids[features[attr] == 1] = np.array(mapping, dtype=np.uint8)
            else:
                try:
                    mapping = minigrid_object_to_encoding_map[attr]
                    grids[features[attr] == 1] = np.array(mapping, dtype=np.uint8)
                except KeyError:
                    pass

        wall_encoding = np.array(
            minigrid_object_to_encoding_map["wall"], dtype=np.uint8
        )
        padded_grid = np.pad(
            grids,
            ((padding, padding), (padding, padding), (0, 0)),
            "constant",
            constant_values=-1,
        )
        padded_grid = np.where(
            padded_grid == -np.ones(3, dtype=np.uint8), wall_encoding, padded_grid
        )
        return padded_grid

    @staticmethod
    def get_node_features(
        graph: nx.Graph, pattern_shape, node_attributes: list[str] = None, reshape=True
    ) -> tuple[np.ndarray, list[str]]:
        if node_attributes is None:
            # Get node attributes from some node
            node_attributes = list(next(iter(graph.nodes.data()))[1].keys())

        # Get node features
        Fx = []
        for attr in node_attributes:
            if attr == "non_navigable" or attr == "wall":
                # The graph we are getting is only the navigable nodes so those that
                # are not present should be assumed to be walls and non-navigable
                f = np.ones(pattern_shape)
            else:
                f = np.zeros(pattern_shape)
            for node, data in graph.nodes.data(attr):
                f[node] = data
            if reshape:
                f = f.ravel()
            Fx.append(f)
        # Fx = torch.stack(Fx, dim=-1).to(device)
        Fx = np.stack(Fx, axis=-1)

        return Fx, node_attributes

    @staticmethod
    def dense_graph_to_minigrid(
        graph: nx.Graph, shape: tuple[int, int], padding=1
    ) -> np.ndarray:
        pattern_shape = (shape[0] - 2 * padding, shape[1] - 2 * padding)
        features, node_attributes = GraphTransforms.get_node_features(
            graph, pattern_shape, node_attributes=None
        )
        # num_zeros = features[features == 0.0].numel()
        # num_ones = features[features == 1.0].numel()
        num_zeros = (features == 0.0).sum()
        num_ones = (features == 1.0).sum()

        assert num_zeros + num_ones == features.size, "Graph features should be binary"
        features_dict = {}
        for i, key in enumerate(node_attributes):
            features_dict[key] = features[..., i]
        grids = GraphTransforms.graph_features_to_minigrid(
            features_dict, shape=shape, padding=padding
        )

        return grids

    @staticmethod
    def get_edge_layers(
        graph: nx.Graph,
        edge_config: dict[str, EdgeDescriptor],
        node_attr: list[str],
        dim_grid: tuple[int, int],
    ) -> dict[str, nx.Graph]:
        navigable_nodes = ["empty", "start", "goal", "moss"]
        non_navigable_nodes = ["wall", "lava"]
        assert all([isinstance(n, tuple) for n in graph.nodes])
        assert all([len(n) == 2 for n in graph.nodes])

        def partial_grid(graph, nodes, dim_grid):
            non_grid_nodes = [n for n in graph.nodes if n not in nodes]
            g_temp = nx.grid_2d_graph(*dim_grid)
            g_temp.remove_nodes_from(non_grid_nodes)
            g_temp.add_nodes_from(non_grid_nodes)
            g = nx.Graph()
            g.add_nodes_from(graph.nodes(data=True))
            g.add_edges_from(g_temp.edges)
            return g

        def pair_edges(graph, node_types):
            all_nodes = []
            for n_type in node_types:
                all_nodes.append(
                    [n for n, a in graph.nodes.items() if a[n_type] >= 1.0]
                )
            edges = list(product(*all_nodes))
            edged_graph = nx.create_empty_copy(graph, with_data=True)
            edged_graph.add_edges_from(edges)
            return edged_graph

        edge_graphs = {}
        for edge_ in edge_config.keys():
            if edge_ == "navigable" and "navigable" not in node_attr:
                edge_config[edge_].between = navigable_nodes
            elif edge_ == "non_navigable" and "non_navigable" not in node_attr:
                edge_config[edge_].between = non_navigable_nodes
            elif not set(edge_config[edge_].between).issubset(set(node_attr)):
                # TODO: remove
                # logger.warning(f"Edge {edge_} not compatible with node attributes {node_attr}. Skipping.")
                continue
            if edge_config[edge_].structure is None:
                edge_graphs[edge_] = pair_edges(graph, edge_config[edge_].between)
            elif edge_config[edge_].structure == "grid":
                nodes = []
                for n_type in edge_config[edge_].between:
                    nodes += [
                        n
                        for n, a in graph.nodes.items()
                        if a[n_type] >= 1.0 and n not in nodes
                    ]
                edge_graphs[edge_] = partial_grid(graph, nodes, dim_grid)
            else:
                raise NotImplementedError(
                    f"Edge structure {edge_config[edge_].structure} not supported."
                )

        return edge_graphs
