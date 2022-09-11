import math

import gymnasium as gym
import numpy as np
import pytest

from minigrid.core.actions import Actions
from minigrid.envs import EmptyEnv
from minigrid.wrappers import (
    ActionBonus,
    DictObservationSpaceWrapper,
    FlatObsWrapper,
    FullyObsWrapper,
    ImgObsWrapper,
    OneHotPartialObsWrapper,
    ReseedWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    StateBonus,
    ViewSizeWrapper,
)
from tests.utils import all_testing_env_specs, assert_equals

SEEDS = [100, 243, 500]
NUM_STEPS = 100


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_reseed_wrapper(env_spec):
    """
    Test the ReseedWrapper with a list of SEEDS.
    """
    unwrapped_env = env_spec.make()
    env = env_spec.make()
    env = ReseedWrapper(env, seeds=SEEDS)
    env.action_space.seed(0)

    for seed in SEEDS:
        env.reset()
        unwrapped_env.reset(seed=seed)
        for time_step in range(NUM_STEPS):
            action = env.action_space.sample()

            obs, rew, terminated, truncated, info = env.step(action)
            (
                unwrapped_obs,
                unwrapped_rew,
                unwrapped_terminated,
                unwrapped_truncated,
                unwrapped_info,
            ) = unwrapped_env.step(action)

            assert_equals(obs, unwrapped_obs, f"[{time_step}] ")
            assert unwrapped_env.observation_space.contains(obs)

            assert (
                rew == unwrapped_rew
            ), f"[{time_step}] reward={rew}, unwrapped reward={unwrapped_rew}"
            assert (
                terminated == unwrapped_terminated
            ), f"[{time_step}] terminated={terminated}, unwrapped terminated={unwrapped_terminated}"
            assert (
                truncated == unwrapped_truncated
            ), f"[{time_step}] truncated={truncated}, unwrapped truncated={unwrapped_truncated}"
            assert_equals(info, unwrapped_info, f"[{time_step}] ")

            # Start the next seed
            if terminated or truncated:
                break

    env.close()
    unwrapped_env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-Empty-16x16-v0"])
def test_state_bonus_wrapper(env_id):
    env = gym.make(env_id)
    wrapped_env = StateBonus(gym.make(env_id))

    action_forward = Actions.forward
    action_left = Actions.left
    action_right = Actions.right

    for _ in range(10):
        wrapped_env.reset()
        for _ in range(5):
            wrapped_env.step(action_forward)

    # Turn lef 3 times (check that actions don't influence bonus)
    for _ in range(3):
        _, wrapped_rew, _, _, _ = wrapped_env.step(action_left)

    env.reset()
    for _ in range(5):
        env.step(action_forward)
    # Turn right 3 times
    for _ in range(3):
        _, rew, _, _, _ = env.step(action_right)

    expected_bonus_reward = rew + 1 / math.sqrt(13)

    assert expected_bonus_reward == wrapped_rew


@pytest.mark.parametrize("env_id", ["MiniGrid-Empty-16x16-v0"])
def test_action_bonus_wrapper(env_id):
    env = gym.make(env_id)
    wrapped_env = ActionBonus(gym.make(env_id))

    action = Actions.forward

    for _ in range(10):
        wrapped_env.reset()
        for _ in range(5):
            _, wrapped_rew, _, _, _ = wrapped_env.step(action)

    env.reset()
    for _ in range(5):
        _, rew, _, _, _ = env.step(action)

    expected_bonus_reward = rew + 1 / math.sqrt(10)

    assert expected_bonus_reward == wrapped_rew


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_dict_observation_space_wrapper(env_spec):
    env = env_spec.make()
    env = DictObservationSpaceWrapper(env)
    env.reset()
    mission = env.mission
    obs, _, _, _, _ = env.step(0)
    assert env.string_to_indices(mission) == [
        value for value in obs["mission"] if value != 0
    ]
    env.close()


@pytest.mark.parametrize(
    "wrapper",
    [
        ReseedWrapper,
        ImgObsWrapper,
        FlatObsWrapper,
        ViewSizeWrapper,
        DictObservationSpaceWrapper,
        OneHotPartialObsWrapper,
        RGBImgPartialObsWrapper,
        FullyObsWrapper,
    ],
)
@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_main_wrappers(wrapper, env_spec):
    env = env_spec.make()
    env = wrapper(env)
    for _ in range(10):
        env.reset()
        env.step(0)
    env.close()


@pytest.mark.parametrize(
    "wrapper",
    [
        OneHotPartialObsWrapper,
        RGBImgPartialObsWrapper,
        FullyObsWrapper,
    ],
)
@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_observation_space_wrappers(wrapper, env_spec):
    env = wrapper(env_spec.make(disable_env_checker=True))
    obs_space, wrapper_name = env.observation_space, wrapper.__name__
    assert isinstance(
        obs_space, gym.spaces.Dict
    ), f"Observation space for {wrapper_name} is not a Dict: {obs_space}."
    # This should not fail either
    ImgObsWrapper(env)
    env.reset()
    env.step(0)
    env.close()


class EmptyEnvWithExtraObs(EmptyEnv):
    """
    Custom environment with an extra observation
    """

    def __init__(self) -> None:
        super().__init__(size=5)
        self.observation_space["size"] = gym.spaces.Box(
            low=0, high=np.iinfo(np.uint).max, shape=(2,), dtype=np.uint
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs["size"] = np.array([self.width, self.height])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs["size"] = np.array([self.width, self.height])
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize(
    "wrapper",
    [
        OneHotPartialObsWrapper,
        RGBImgObsWrapper,
        RGBImgPartialObsWrapper,
        FullyObsWrapper,
    ],
)
def test_agent_sees_method(wrapper):
    env1 = wrapper(EmptyEnvWithExtraObs())
    env2 = wrapper(gym.make("MiniGrid-Empty-5x5-v0"))

    obs1, _ = env1.reset(seed=0)
    obs2, _ = env2.reset(seed=0)
    assert "size" in obs1
    assert obs1["size"].shape == (2,)
    assert (obs1["size"] == [5, 5]).all()
    for key in obs2:
        assert np.array_equal(obs1[key], obs2[key])

    obs1, reward1, terminated1, truncated1, _ = env1.step(0)
    obs2, reward2, terminated2, truncated2, _ = env2.step(0)
    assert "size" in obs1
    assert obs1["size"].shape == (2,)
    assert (obs1["size"] == [5, 5]).all()
    for key in obs2:
        assert np.array_equal(obs1[key], obs2[key])
