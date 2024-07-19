from __future__ import annotations

import copy

import networkx as nx
import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.envs.wfc.config import WFC_PRESETS_ALL, WFCConfig
from minigrid.envs.wfc.graphtransforms import EdgeDescriptor, GraphTransforms
from minigrid.envs.wfc.wfclogic.control import execute_wfc
from minigrid.minigrid_env import MiniGridEnv

FEATURE_DESCRIPTORS = {"empty", "wall", "lava", "start", "goal"} | {
    "navigable",
    "non_navigable",
}

EDGE_CONFIG = {
    "navigable": EdgeDescriptor(between=("navigable",), structure="grid"),
    "non_navigable": EdgeDescriptor(between=("non_navigable",), structure="grid"),
    "start_goal": EdgeDescriptor(between=("start", "goal"), structure=None),
    # "lava_goal": EdgeDescriptor(between=("lava", "goal"), weight="lava_prob"),
    # "moss_goal": EdgeDescriptor(between=("moss", "goal"), weight="moss_prob"),
}


class WFCEnv(MiniGridEnv):
    """
    ## Description

    This environment procedurally generates a level using the Wave Function Collapse algorithm.
    The environment supports a variety of different level structures but the default is a simple maze.
    See [WFC module page](index) for sample images of the available presets.

    Requires the optional dependencies `imageio` and `networkx` to be installed with `pip install minigrid[wfc]`.

    ## Mission Space

    "traverse the maze to get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Unused                    |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked (unused)

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-WFC-MazeSimple-v0`
    - `MiniGrid-WFC-DungeonMazeScaled-v0`
    - `MiniGrid-WFC-RoomsFabric-v0`
    - `MiniGrid-WFC-ObstaclesBlackdots-v0`
    - `MiniGrid-WFC-ObstaclesAngular-v0`
    - `MiniGrid-WFC-ObstaclesHogs3-v0`

    Note: There are many more unregistered configuration presets but they may take a long time to generate a consistent environment.

    They can be registered with the following snippet:
    ```python
    import gymnasium
    from minigrid.envs.wfc.config import register_wfc_presets, WFC_PRESETS_INCONSISTENT, WFC_PRESETS_SLOW

    register_wfc_presets(WFC_PRESETS_INCONSISTENT, gymnasium.register)
    register_wfc_presets(WFC_PRESETS_SLOW, gymnasium.register)
    ```

    ## Research

    Adapted for `Minigrid` by the following work.

    ```bibtex
    @inproceedings{garcin2024dred,
      title = {DRED: Zero-Shot Transfer in Reinforcement Learning via Data-Regularised Environment Design},
      author = {Garcin, Samuel and Doran, James and Guo, Shangmin and Lucas, Christopher G and Albrecht, Stefano V},
      booktitle = {Forty-first International Conference on Machine Learning},
      year = {2024},
    }
    ```
    """

    PATTERN_COLOR_CONFIG = {
        "wall": (0, 0, 0),  # black
        "empty": (255, 255, 255),  # white
    }

    def __init__(
        self,
        wfc_config: WFCConfig | str = "MazeSimple",
        size: int = 25,
        ensure_connected: bool = True,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.config = (
            wfc_config
            if isinstance(wfc_config, WFCConfig)
            else WFC_PRESETS_ALL[wfc_config]
        )
        self.padding = 1

        # This controls whether to process the level such that there is only a single connected navigable area
        self.ensure_connected = ensure_connected

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if size < 3:
            raise ValueError(f"Grid size must be at least 3 (currently {size})")
        self.size = size
        self.max_attempts = 1000

        if max_steps is None:
            max_steps = self.size * 20

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "traverse the maze to get to the goal"

    def _gen_grid(self, width, height):
        shape = (height, width)

        # Main call to generate a black and white pattern with WFC
        shape_unpadded = (shape[0] - 2 * self.padding, shape[1] - 2 * self.padding)
        pattern, _stats = execute_wfc(
            attempt_limit=self.max_attempts,
            output_size=shape_unpadded,
            np_random=self.np_random,
            **self.config.wfc_kwargs,
        )
        if pattern is None:
            raise RuntimeError(
                f"Could not generate a valid pattern within {self.max_attempts} attempts"
            )

        grid_raw = self._pattern_to_minigrid_layout(pattern)

        # Stage 1: Make a navigable graph with only one main cavern
        stage1_edge_config = {k: v for k, v in EDGE_CONFIG.items() if k == "navigable"}
        graph_raw, _edge_graphs = GraphTransforms.minigrid_layout_to_dense_graph(
            grid_raw[np.newaxis],
            remove_border=False,
            node_attr=FEATURE_DESCRIPTORS,
            edge_config=stage1_edge_config,
        )
        graph = graph_raw[0]

        # Stage 2: Graph processing
        # Retain only the largest connected graph component, fill in the rest with walls
        if self.ensure_connected:
            graph = self._get_largest_component(graph)

        # Add start and goal nodes
        graph = self._place_start_and_goal_random(graph)

        # Convert graph back to grid
        grid_array = GraphTransforms.dense_graph_to_minigrid(
            graph, shape=shape, padding=self.padding
        )

        # Decode to minigrid and set variables
        self.agent_dir = self._rand_int(0, 4)
        self.agent_pos = next(
            zip(*np.nonzero(grid_array[:, :, 0] == OBJECT_TO_IDX["agent"]))
        )
        self.grid, _vismask = Grid.decode(grid_array)
        self.mission = self._gen_mission()

    def _pattern_to_minigrid_layout(self, pattern: np.ndarray):
        if pattern.ndim != 3:
            raise ValueError(
                f"Expected pattern to have 3 dimensions, but got {pattern.ndim}"
            )
        layout = np.ones(pattern.shape, dtype=np.uint8) * OBJECT_TO_IDX["empty"]

        wall_ids = np.where(pattern == self.PATTERN_COLOR_CONFIG["wall"])
        layout[wall_ids] = OBJECT_TO_IDX["wall"]
        layout = layout[..., 0]

        return layout

    @staticmethod
    def _get_largest_component(graph: nx.Graph) -> nx.Graph:
        wall_graph_attr = GraphTransforms.OBJECT_TO_DENSE_GRAPH_ATTRIBUTE["wall"]
        # Prepare graph
        inactive_nodes = [x for x, y in graph.nodes(data=True) if y["navigable"] < 0.5]
        graph.remove_nodes_from(inactive_nodes)

        components = [
            graph.subgraph(c).copy()
            for c in sorted(nx.connected_components(graph), key=len, reverse=True)
            if len(c) > 1
        ]
        component = components[0]
        graph = graph.subgraph(component)

        for node in graph.nodes():
            if node not in component.nodes():
                for feat in graph.nodes[node]:
                    if feat in wall_graph_attr:
                        graph.nodes[node][feat] = 1.0
                    else:
                        graph.nodes[node][feat] = 0.0
        # TODO: Check if this is necessary
        g = nx.Graph()
        g.add_nodes_from(graph.nodes(data=True))
        g.add_edges_from(component.edges(data=True))

        g_out = copy.deepcopy(g)

        return g_out

    def _place_start_and_goal_random(self, graph: nx.Graph) -> nx.Graph:
        node_set = "navigable"

        # Get two random navigable nodes
        possible_nodes = [n for n, d in graph.nodes(data=True) if d[node_set]]
        inds = self.np_random.permutation(len(possible_nodes))[:2]
        start_node, goal_node = possible_nodes[inds[0]], possible_nodes[inds[1]]

        graph.nodes[start_node]["start"] = 1
        graph.nodes[goal_node]["goal"] = 1

        return graph
