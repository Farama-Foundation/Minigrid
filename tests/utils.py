"""Finds all the specs that we can test with"""

from __future__ import annotations

from importlib.util import find_spec

import gymnasium as gym
import numpy as np

all_testing_env_specs = [
    env_spec
    for env_spec in gym.envs.registry.values()
    if (
        isinstance(env_spec.entry_point, str)
        and env_spec.entry_point.startswith("minigrid.envs")
    )
]

if find_spec("imageio") is None or find_spec("networkx") is None:
    # Do not test WFC environments if dependencies are not installed
    all_testing_env_specs = [
        env_spec
        for env_spec in all_testing_env_specs
        if not env_spec.entry_point.startswith("minigrid.envs.wfc")
    ]

minigrid_testing_env_specs = [
    env_spec
    for env_spec in all_testing_env_specs
    if not env_spec.entry_point.startswith("minigrid.envs.babyai")
]


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.

    Args:
        a: first data structure
        b: second data structure
        prefix: prefix for failed assertion message for types and dicts
    """
    assert type(a) is type(b), f"{prefix}Differing types: {a} and {b}"
    if isinstance(a, dict):
        assert list(a.keys()) == list(b.keys()), f"{prefix}Key sets differ: {a} and {b}"

        for k in a.keys():
            v_a = a[k]
            v_b = b[k]
            assert_equals(v_a, v_b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple):
        for elem_from_a, elem_from_b in zip(a, b):
            assert_equals(elem_from_a, elem_from_b)
    else:
        assert a == b
