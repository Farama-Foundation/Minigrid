"""Experimental multi-agent navigation environments for issue #514.

The task families follow the environments proposed for adoption from
https://github.com/kandouss/marlgrid. This is a task-level reimplementation using
modern Minigrid objects and PettingZoo, not a drop-in replacement for marlgrid.
Install ``minigrid[multiagent]`` to construct the environments.
"""

from __future__ import annotations

from minigrid.multiagent.compat import MarlgridListAdapter, make_marlgrid
from minigrid.multiagent.env import ClutteredParallelEnv, EmptyParallelEnv
from minigrid.multiagent.tasks import (
    DoorKeyParallelEnv,
    GoalCycleParallelEnv,
    VisibilityParallelEnv,
)

__all__ = [
    "ClutteredParallelEnv",
    "EmptyParallelEnv",
    "DoorKeyParallelEnv",
    "GoalCycleParallelEnv",
    "VisibilityParallelEnv",
    "MarlgridListAdapter",
    "make_marlgrid",
]
