---
firstpage:
lastpage:
---

## WFC Environments

The environments listed below are implemented in the [minigrid/envs/wfc](/minigrid/envs/wfc)
directory. Instead of laying out a level from fixed rules, each one generates it with the
[Wave Function Collapse algorithm](https://github.com/mxgmn/WaveFunctionCollapse) - adapted from the
[Python implementation by Isaac Karth](https://github.com/ikarth/wfc_2019f) - which samples a new
grid that is locally similar to a small example image, so every `reset()` gives a different level of
the same character. The mission is always to traverse the level to the goal; what changes between
the environments is the pattern they sample from, one per preset, giving mazes, dungeons, rooms and
scattered obstacles. These environments need the optional dependencies, installed with
`pip install minigrid[wfc]`.

```{raw} html
   :file: list.html
```

```{toctree}
:hidden:
:caption: Wave Function Collapse Environments

WFCEnv
WFCMazeSimpleEnv
WFCDungeonMazeScaledEnv
WFCRoomsFabricEnv
WFCObstaclesBlackdotsEnv
WFCObstaclesAngularEnv
WFCObstaclesHogs3Env
WFCMazeKnotEnv
WFCMazeWallEnv
WFCRoomsOfficeEnv
WFCObstaclesHogs2Env
WFCSkew2Env
WFCMazeEnv
WFCMazeSpiralsEnv
WFCMazePathsEnv
WFCMazelikeEnv
WFCDungeonEnv
WFCDungeonRoomsEnv
WFCDungeonLessRoomsEnv
WFCDungeonSpiralsEnv
WFCRoomsMagicOfficeEnv
WFCSkewCaveEnv
WFCSkewLakeEnv

```
