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
        docstring: Documentation used for the generated environment page.
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
    docstring: str | None = None

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
        kwargs.pop("docstring")
        kwargs["image"] = imread(kwargs.pop("pattern_path"))[:, :, :3]
        return kwargs


_WFC_COMMON_DOC = """
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

## Dependencies

Requires the optional dependencies `imageio` and `networkx` to be installed with `pip install minigrid[wfc]`.

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


def _wfc_docstring(
    preset_name: str,
    description: str,
    *,
    registered_by_default: bool = True,
    registration_group: str | None = None,
    slow: bool = False,
    generation_note: str = "This preset is intended to generate in under a minute.",
):
    if registered_by_default:
        registration_note = (
            "This preset is registered by default and can be created directly with "
            f'`gymnasium.make("MiniGrid-WFC-{preset_name}-v0")`.'
        )
        registration_snippet = ""
    else:
        registration_note = (
            "This preset is not registered by default. Register the additional WFC "
            "preset group before creating it."
        )
        registration_snippet = f"""
```python
import gymnasium
from minigrid.envs.wfc.config import {registration_group}, register_wfc_presets

register_wfc_presets({registration_group}, gymnasium.register)
```
"""

    return f"""
## Description

This environment procedurally generates a level using the Wave Function Collapse algorithm.
The `{preset_name}` preset {description}

See [WFC module page](index) for sample images of the available presets.

## WFC Preset

|   |   |
|---|---|
| Preset | `{preset_name}` |
| Registered by default | {"Yes" if registered_by_default else "No"} |
| Requires additional registration | {"No" if registered_by_default else "Yes"} |
| Slow preset | {"Yes" if slow else "No"} |

## Registration

{registration_note}
{registration_snippet}

## Generation Notes

{generation_note}

{_WFC_COMMON_DOC}
"""


# Basic presets for WFC configurations (that should generate in <1 min)
WFC_PRESETS = {
    "MazeSimple": WFCConfig(
        pattern_path=PATTERN_PATH / "SimpleMaze.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=False,
        input_periodic=False,
        docstring=_wfc_docstring(
            "MazeSimple",
            "learns from a compact simple-maze pattern to create sparse corridors "
            "and wall-separated passages.",
        ),
    ),
    "DungeonMazeScaled": WFCConfig(
        pattern_path=PATTERN_PATH / "ScaledMaze.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "DungeonMazeScaled",
            "learns from a scaled dungeon-maze pattern to create larger repeating "
            "corridor structures.",
        ),
    ),
    "RoomsFabric": WFCConfig(
        pattern_path=PATTERN_PATH / "Fabric.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=False,
        input_periodic=False,
        docstring=_wfc_docstring(
            "RoomsFabric",
            "learns from a fabric-like room pattern to create blocky connected "
            "spaces with repeated interior texture.",
        ),
    ),
    "ObstaclesBlackdots": WFCConfig(
        pattern_path=PATTERN_PATH / "Blackdots.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=False,
        input_periodic=False,
        docstring=_wfc_docstring(
            "ObstaclesBlackdots",
            "learns from a black-dot obstacle pattern to scatter small wall "
            "clusters through otherwise open space.",
        ),
    ),
    "ObstaclesAngular": WFCConfig(
        pattern_path=PATTERN_PATH / "Angular.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "ObstaclesAngular",
            "learns from an angular obstacle pattern to create diagonal-looking "
            "barriers and jagged room boundaries.",
        ),
    ),
    "ObstaclesHogs3": WFCConfig(
        pattern_path=PATTERN_PATH / "Hogs.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "ObstaclesHogs3",
            "learns width-3 patterns from the Hogs source image to create dense "
            "organic obstacle fields.",
        ),
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
        docstring=_wfc_docstring(
            "MazeKnot",
            "learns from a knot-like maze pattern to create tight, tangled "
            "corridor structures.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_INCONSISTENT",
            generation_note="This preset is not marked as slow, but it can require many attempts to generate a consistent level.",
        ),
    ),  # This is not too inconsistent (often 10 attempts is enough)
    "MazeWall": WFCConfig(
        pattern_path=PATTERN_PATH / "SimpleWall.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "MazeWall",
            "learns from a simple wall pattern to create heavier maze barriers "
            "with repeated wall segments.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_INCONSISTENT",
            generation_note="This preset is not marked as slow, but it can require many attempts to generate a consistent level.",
        ),
    ),
    "RoomsOffice": WFCConfig(
        pattern_path=PATTERN_PATH / "Office.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "RoomsOffice",
            "learns from an office-like room pattern to create rectilinear rooms, "
            "hallways, and partitions.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_INCONSISTENT",
            generation_note="This preset is not marked as slow, but it can require many attempts to generate a consistent level.",
        ),
    ),
    "ObstaclesHogs2": WFCConfig(
        pattern_path=PATTERN_PATH / "Hogs.png",
        tile_size=1,
        pattern_width=2,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "ObstaclesHogs2",
            "learns width-2 patterns from the Hogs source image to create smaller "
            "organic obstacle clusters.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_INCONSISTENT",
            generation_note="This preset is not marked as slow, but it can require many attempts to generate a consistent level.",
        ),
    ),
    "Skew2": WFCConfig(
        pattern_path=PATTERN_PATH / "Skew2.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "Skew2",
            "learns from a skewed pattern to create asymmetric cave-like layouts "
            "with angled wall contours.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_INCONSISTENT",
            generation_note="This preset is not marked as slow, but it can require many attempts to generate a consistent level.",
        ),
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
        docstring=_wfc_docstring(
            "Maze",
            "learns from a larger maze pattern to create complex corridor networks.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),  # This is unusually slow: ~20min per 25x25 room
    "MazeSpirals": WFCConfig(
        pattern_path=PATTERN_PATH / "Spirals.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "MazeSpirals",
            "learns from spiral motifs to create curling corridors and rounded "
            "maze-like structures.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),
    "MazePaths": WFCConfig(
        pattern_path=PATTERN_PATH / "Paths.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "MazePaths",
            "learns from a path-heavy maze pattern to create branching walkways.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),
    "Mazelike": WFCConfig(
        pattern_path=PATTERN_PATH / "Mazelike.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "Mazelike",
            "learns from a maze-like pattern to create irregular wall structures "
            "and navigable channels.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),
    "Dungeon": WFCConfig(
        pattern_path=PATTERN_PATH / "DungeonExtr.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "Dungeon",
            "learns from an extracted dungeon pattern to create dense dungeon "
            "layouts with rooms and passages.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),  # ~10 mins
    "DungeonRooms": WFCConfig(
        pattern_path=PATTERN_PATH / "Rooms.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "DungeonRooms",
            "learns from a room-focused dungeon pattern to create larger chambered "
            "layouts.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),
    "DungeonLessRooms": WFCConfig(
        pattern_path=PATTERN_PATH / "LessRooms.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "DungeonLessRooms",
            "learns from a sparse dungeon-room pattern to create layouts with fewer "
            "large chambers and more dividing walls.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),
    "DungeonSpirals": WFCConfig(
        pattern_path=PATTERN_PATH / "SpiralsNeg.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "DungeonSpirals",
            "learns from a negative spiral pattern to create dungeon layouts with "
            "curved-looking passages.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),
    "RoomsMagicOffice": WFCConfig(
        pattern_path=PATTERN_PATH / "MagicOffice.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "RoomsMagicOffice",
            "learns from a magic-office pattern to create partitioned room layouts "
            "with varied interior structure.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),
    "SkewCave": WFCConfig(
        pattern_path=PATTERN_PATH / "Cave.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=False,
        input_periodic=False,
        docstring=_wfc_docstring(
            "SkewCave",
            "learns from a cave pattern to create skewed cavern shapes and uneven "
            "walls.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),
    "SkewLake": WFCConfig(
        pattern_path=PATTERN_PATH / "Lake.png",
        tile_size=1,
        pattern_width=3,
        output_periodic=True,
        input_periodic=True,
        docstring=_wfc_docstring(
            "SkewLake",
            "learns from a lake pattern to create skewed open regions enclosed by "
            "irregular walls.",
            registered_by_default=False,
            registration_group="WFC_PRESETS_SLOW",
            slow=True,
            generation_note="This preset is slow and can take several minutes to generate.",
        ),
    ),  # ~10 mins
}

WFC_PRESETS_ALL = ChainMap(WFC_PRESETS, WFC_PRESETS_INCONSISTENT, WFC_PRESETS_SLOW)


def _wfc_env_class_name(preset_name: str) -> str:
    return f"WFC{preset_name}Env"


def _ensure_wfc_env_class(preset_name: str, config: WFCConfig) -> str:
    from minigrid.envs import wfc as wfc_module

    class_name = _wfc_env_class_name(preset_name)
    if not hasattr(wfc_module, class_name):
        docstring = config.docstring or _wfc_docstring(
            preset_name,
            "uses its configured source pattern to create procedural layouts.",
        )
        preset_env = type(
            class_name,
            (wfc_module.WFCEnv,),
            {
                "__doc__": docstring,
                "__module__": "minigrid.envs.wfc",
            },
        )
        setattr(wfc_module, class_name, preset_env)
    return class_name


def register_wfc_presets(wfc_presets: dict[str, WFCConfig], register_fn):
    # Register fn needs to be provided to avoid a circular import
    for name, config in wfc_presets.items():
        class_name = _ensure_wfc_env_class(name, config)
        register_fn(
            id=f"MiniGrid-WFC-{name}-v0",
            entry_point=f"minigrid.envs.wfc:{class_name}",
            kwargs={"wfc_config": name},
        )
