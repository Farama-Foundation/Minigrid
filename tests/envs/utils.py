"""Finds all the specs that we can test with"""
from typing import Optional

import gym
import numpy as np
from gym import logger
from gym.envs.registration import EnvSpec


def try_make_env(env_spec: EnvSpec) -> Optional[gym.Env]:
    """Tries to make the environment showing if it is possible. Warning the environments have no wrappers, including time limit and order enforcing."""
    try:
        return env_spec.make(disable_env_checker=True).unwrapped
    except ImportError as e:
        logger.warn(f"Not testing {env_spec.id} due to error: {e}")
        return None


# Tries to make all gym_minigrid environment to test with
all_testing_initialised_envs = list(
    filter(
        None,
        [
            try_make_env(env_spec)
            for env_spec in gym.envs.registry.values()
            if env_spec.entry_point.startswith("gym_minigrid.envs")
        ],
    )
)
all_testing_env_specs = [env.spec for env in all_testing_initialised_envs]


def assert_equals(a, b, prefix=None):
    """Assert equality of data structures `a` and `b`.

    Args:
        a: first data structure
        b: second data structure
        prefix: prefix for failed assertion message for types and dicts
    """
    assert type(a) == type(b), f"{prefix}Differing types: {a} and {b}"
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
