from __future__ import annotations

from collections import ChainMap
from dataclasses import asdict, dataclass
from pathlib import Path

from typing_extensions import Literal

PATTERN_PATH = Path(__file__).parent / "patterns"


@dataclass
class WFCConfig:
    """Dataclass for holding WFC configuration parameters.

    This controls the behavior of the WFC algorithm. The parameters are passed directly to the WFC solver.

    Attributes:
        pattern_path: Path to the pattern image that will be automatically loaded.
        tile_size: Size of the tiles in pixels to create from the pattern image.
        pattern_width: Size of the patterns in tiles to take from the pattern image. (greater than 3 is quite slow)
        rotations: Number of rotations for each tile.
        output_periodic: Whether the output should be periodic (wraps over edges).
        input_periodic: Whether the input should be periodic (wraps over edges).
        loc_heuristic: Heuristic for choosing the next tile location to collapse.
        choice_heuristic: Heuristic for choosing the next tile to use between possible tiles.
        backtracking: Whether to backtrack when contradictions are discovered.
    """

    pattern_path: Path
    tile_size: int = 1
    pattern_width: int = 2
    rotations: int = 8
    output_periodic: bool = False
    input_periodic: bool = False
    loc_heuristic: Literal[
        "lexical", "spiral", "entropy", "anti-entropy", "simple", "random"
    ] = "entropy"
    choice_heuristic: Literal["lexical", "rarest", "weighted", "random"] = "weighted"
    backtracking: bool = False

    @property
    def wfc_kwargs(self):
        try:
            from imageio.v2 import imread
        except ImportError as e:
            from gymnasium.error import DependencyNotInstalled

            raise DependencyNotInstalled(
                'imageio is missing, please run `pip install "minigrid[wfc]"`'
            ) from e
        kwargs = asdict(self)
        kwargs["image"] = imread(kwargs.pop("pattern_path"))[:, :, :3]
        return kwargs


# Basic presets for WFC configurations (that should generate in <1 min)
WFC_PRESETS = {
    "MazeSimple": WFCConfig(
        pattern_path=PATTERN_PATH / "SimpleMaze.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=False,
        input_periodic=False,
    ),
    "DungeonMazeScaled": WFCConfig(
        pattern_path=PATTERN_PATH / "ScaledMaze.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
    ),
    "RoomsFabric": WFCConfig(
        pattern_path=PATTERN_PATH / "Fabric.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=False,
        input_periodic=False,
    ),
    "ObstaclesBlackdots": WFCConfig(
        pattern_path=PATTERN_PATH / "Blackdots.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=False,
        input_periodic=False,
    ),
    "ObstaclesAngular": WFCConfig(
        pattern_path=PATTERN_PATH / "Angular.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "ObstaclesHogs3": WFCConfig(
        pattern_path=PATTERN_PATH / "Hogs.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
}

# Presets that take a large number of attempts to generate a consistent environment
WFC_PRESETS_INCONSISTENT = {
    "MazeKnot": WFCConfig(
        pattern_path=PATTERN_PATH / "Knot.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),  # This is not too inconsistent (often 10 attempts is enough)
    "MazeWall": WFCConfig(
        pattern_path=PATTERN_PATH / "SimpleWall.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
    ),
    "RoomsOffice": WFCConfig(
        pattern_path=PATTERN_PATH / "Office.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "ObstaclesHogs2": WFCConfig(
        pattern_path=PATTERN_PATH / "Hogs.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
    ),
    "Skew2": WFCConfig(
        pattern_path=PATTERN_PATH / "Skew2.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
}

# Slow presets for WFC configurations (Most take about 2-4 min but some take 10+ min)
WFC_PRESETS_SLOW = {
    "Maze": WFCConfig(
        pattern_path=PATTERN_PATH / "Maze.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),  # This is unusually slow: ~20min per 25x25 room
    "MazeSpirals": WFCConfig(
        pattern_path=PATTERN_PATH / "Spirals.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "MazePaths": WFCConfig(
        pattern_path=PATTERN_PATH / "Paths.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "Mazelike": WFCConfig(
        pattern_path=PATTERN_PATH / "Mazelike.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "Dungeon": WFCConfig(
        pattern_path=PATTERN_PATH / "DungeonExtr.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),  # ~10 mins
    "DungeonRooms": WFCConfig(
        pattern_path=PATTERN_PATH / "Rooms.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "DungeonLessRooms": WFCConfig(
        pattern_path=PATTERN_PATH / "LessRooms.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "DungeonSpirals": WFCConfig(
        pattern_path=PATTERN_PATH / "SpiralsNeg.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "RoomsMagicOffice": WFCConfig(
        pattern_path=PATTERN_PATH / "MagicOffice.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),
    "SkewCave": WFCConfig(
        pattern_path=PATTERN_PATH / "Cave.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=False,
        input_periodic=False,
    ),
    "SkewLake": WFCConfig(
        pattern_path=PATTERN_PATH / "Lake.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
    ),  # ~10 mins
}

WFC_PRESETS_ALL = ChainMap(WFC_PRESETS, WFC_PRESETS_INCONSISTENT, WFC_PRESETS_SLOW)


def register_wfc_presets(wfc_presets: dict[str, WFCConfig], register_fn):
    # Register fn needs to be provided to avoid a circular import
    for name in wfc_presets.keys():
        register_fn(
            id=f"MiniGrid-WFC-{name}-v0",
            entry_point="minigrid.envs.wfc:WFCEnv",
            kwargs={"wfc_config": name},
        )
