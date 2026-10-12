"""Execute pinned upstream sources headlessly with dependency-only shims.

This is not a native legacy dependency installation. No transition, placement,
reward, or termination method is replaced. NumPy aliases, Gym's RNG method name,
rendering imports and the optional visibility JIT are adapted for Python 3.13.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import gymnasium
import numpy as np

SOURCE = Path(os.environ.get("MARLGRID_SOURCE", "marlgrid-source")).resolve()
SOURCE_COMMIT = "e88c40bad07653575ac11fe2f3a115e4de3d13e9"


class LegacyRNG:
    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)

    def randint(self, low, high=None, size=None):
        return self.generator.integers(low, high, size)

    def __getattr__(self, name):
        return getattr(self.generator, name)


def load_reference():
    from minigrid.utils import rendering

    if "_marlgrid_source.base" in sys.modules:
        return tuple(
            importlib.import_module("_marlgrid_source." + name)
            for name in ("base", "agents", "objects", "envs.empty", "envs.cluttered")
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    if revision != SOURCE_COMMIT or changed:
        raise RuntimeError(
            "Reference capture requires the clean pinned marlgrid revision"
        )
    for name, alias in (("int", int), ("float", float), ("bool", np.bool_)):
        if name not in np.__dict__:
            setattr(np, name, alias)
    gym = types.ModuleType("gym")
    gym.Env = gymnasium.Env
    gym.spaces = gymnasium.spaces
    gym.utils = types.SimpleNamespace(
        seeding=types.SimpleNamespace(np_random=lambda seed: (LegacyRNG(seed), seed))
    )
    sys.modules["gym"] = gym
    old_minigrid = types.ModuleType("gym_minigrid")
    old_minigrid.rendering = rendering
    sys.modules["gym_minigrid"] = old_minigrid
    sys.modules["gym_minigrid.rendering"] = rendering
    numba = types.ModuleType("numba")
    numba.njit = lambda function: function
    numba.boolean = np.bool_
    sys.modules["numba"] = numba
    package = types.ModuleType("_marlgrid_source")
    package.__path__ = [str(SOURCE / "marlgrid")]
    sys.modules[package.__name__] = package
    envs = types.ModuleType("_marlgrid_source.envs")
    envs.__path__ = [str(SOURCE / "marlgrid/envs")]
    sys.modules[envs.__name__] = envs
    window = types.ModuleType("_marlgrid_source.rendering")

    class UnsupportedViewer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Only headless reference execution is supported")

    window.SimpleImageViewer = UnsupportedViewer
    sys.modules[window.__name__] = window
    return load_reference_modules()


def load_reference_modules():
    return tuple(
        importlib.import_module("_marlgrid_source." + name)
        for name in ("base", "agents", "objects", "envs.empty", "envs.cluttered")
    )
