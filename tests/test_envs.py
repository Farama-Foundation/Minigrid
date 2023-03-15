from __future__ import annotations

import pickle
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env, data_equivalence

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from tests.utils import all_testing_env_specs, assert_equals

CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_env(spec):
    # Capture warnings
    env = spec.make(disable_env_checker=True).unwrapped
    warnings.simplefilter("always")
    # Test if env adheres to Gym API
    with warnings.catch_warnings(record=True) as w:
        check_env(env)
    for warning in w:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")


# Note that this precludes running this test in multiple threads.
# However, we probably already can't do multithreading due to some environments.
SEED = 0
NUM_STEPS = 50


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[env.id for env in all_testing_env_specs]
)
def test_env_determinism_rollout(env_spec: EnvSpec):
    """Run a rollout with two environments and assert equality.

    This test run a rollout of NUM_STEPS steps with two environments
    initialized with the same seed and assert that:

    - observation after first reset are the same
    - same actions are sampled by the two envs
    - observations are contained in the observation space
    - obs, rew, terminated, truncated and info are equals between the two envs
    """
    # Don't check rollout equality if it's a nondeterministic environment.
    if env_spec.nondeterministic is True:
        return

    env_1 = env_spec.make(disable_env_checker=True)
    env_2 = env_spec.make(disable_env_checker=True)

    initial_obs_1 = env_1.reset(seed=SEED)
    initial_obs_2 = env_2.reset(seed=SEED)
    assert_equals(initial_obs_1, initial_obs_2)

    env_1.action_space.seed(SEED)

    for time_step in range(NUM_STEPS):
        # We don't evaluate the determinism of actions
        action = env_1.action_space.sample()

        obs_1, rew_1, terminated_1, truncated_1, info_1 = env_1.step(action)
        obs_2, rew_2, terminated_2, truncated_2, info_2 = env_2.step(action)

        assert_equals(obs_1, obs_2, f"[{time_step}] ")
        assert env_1.observation_space.contains(
            obs_1
        )  # obs_2 verified by previous assertion

        assert rew_1 == rew_2, f"[{time_step}] reward 1={rew_1}, reward 2={rew_2}"
        assert (
            terminated_1 == terminated_2
        ), f"[{time_step}] terminated 1={terminated_1}, terminated 2={terminated_2}"
        assert (
            truncated_1 == truncated_2
        ), f"[{time_step}] truncated 1={truncated_1}, truncated 2={truncated_2}"
        assert_equals(info_1, info_2, f"[{time_step}] ")

        if (
            terminated_1 or truncated_1
        ):  # terminated_2 and truncated_2 verified by previous assertion
            env_1.reset(seed=SEED)
            env_2.reset(seed=SEED)

    env_1.close()
    env_2.close()


@pytest.mark.parametrize(
    "spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_render_modes(spec):
    env = spec.make()

    for mode in env.metadata.get("render_modes", []):
        if mode != "human":
            new_env = spec.make(render_mode=mode)

            new_env.reset()
            new_env.step(new_env.action_space.sample())
            new_env.render()


@pytest.mark.parametrize("env_id", ["MiniGrid-DoorKey-6x6-v0"])
def test_agent_sees_method(env_id):
    env = gym.make(env_id)
    goal_pos = (env.grid.width - 2, env.grid.height - 2)

    # Test the env.agent_sees() function
    env.reset()
    # Test the "in" operator on grid objects
    assert ("green", "goal") in env.grid
    assert ("blue", "key") not in env.grid
    for i in range(0, 500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        grid, _ = Grid.decode(obs["image"])
        goal_visible = ("green", "goal") in grid

        agent_sees_goal = env.agent_sees(*goal_pos)
        assert agent_sees_goal == goal_visible
        if terminated or truncated:
            env.reset()

    env.close()


@pytest.mark.parametrize(
    "env_spec", all_testing_env_specs, ids=[spec.id for spec in all_testing_env_specs]
)
def test_max_steps_argument(env_spec):
    """
    Test that when initializing an environment with a fixed number of steps per episode (`max_steps` argument),
    the episode will be truncated after taking that number of steps.
    """
    max_steps = 50
    env = env_spec.make(max_steps=max_steps)
    env.reset()
    step_count = 0
    while True:
        _, _, terminated, truncated, _ = env.step(4)
        step_count += 1
        if truncated:
            assert step_count == max_steps
            step_count = 0
            break

    env.close()


@pytest.mark.parametrize(
    "env_spec",
    all_testing_env_specs,
    ids=[spec.id for spec in all_testing_env_specs],
)
def test_pickle_env(env_spec):
    """Test that all environments are picklable."""
    env: gym.Env = env_spec.make()
    pickled_env: gym.Env = pickle.loads(pickle.dumps(env))

    data_equivalence(env.reset(), pickled_env.reset())

    action = env.action_space.sample()
    data_equivalence(env.step(action), pickled_env.step(action))
    env.close()
    pickled_env.close()


@pytest.mark.parametrize(
    "env_spec",
    all_testing_env_specs,
    ids=[spec.id for spec in all_testing_env_specs],
)
def old_run_test(env_spec):
    # Load the gym environment
    env = env_spec.make()
    env.max_steps = min(env.max_steps, 200)
    env.reset()
    env.render()

    # Verify that the same seed always produces the same environment
    for i in range(0, 5):
        seed = 1337 + i
        _ = env.reset(seed=seed)
        grid1 = env.grid
        _ = env.reset(seed=seed)
        grid2 = env.grid
        assert grid1 == grid2

    env.reset()

    # Run for a few episodes
    num_episodes = 0
    while num_episodes < 5:
        # Pick a random action
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        # Validate the agent position
        assert env.agent_pos[0] < env.width
        assert env.agent_pos[1] < env.height

        # Test observation encode/decode roundtrip
        img = obs["image"]
        grid, vis_mask = Grid.decode(img)
        img2 = grid.encode(vis_mask=vis_mask)
        assert np.array_equal(img, img2)

        # Test the env to string function
        str(env)

        # Check that the reward is within the specified range
        assert reward >= env.reward_range[0], reward
        assert reward <= env.reward_range[1], reward

        if terminated or truncated:
            num_episodes += 1
            env.reset()

        env.render()

    # Test the close method
    env.close()


@pytest.mark.parametrize("env_id", ["MiniGrid-Empty-8x8-v0"])
def test_interactive_mode(env_id):
    env = gym.make(env_id)
    env.reset()

    for i in range(0, 100):
        print(f"step {i}")

        # Pick a random action
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

    # Test the close method
    env.close()


def test_mission_space():

    # Test placeholders
    mission_space = MissionSpace(
        mission_func=lambda color, obj_type: f"Get the {color} {obj_type}.",
        ordered_placeholders=[["green", "red"], ["ball", "key"]],
    )

    assert mission_space.contains("Get the green ball.")
    assert mission_space.contains("Get the red key.")
    assert not mission_space.contains("Get the purple box.")

    # Test passing inverted placeholders
    assert not mission_space.contains("Get the key red.")

    # Test passing extra repeated placeholders
    assert not mission_space.contains("Get the key red key.")

    # Test contained placeholders like "get the" and "go get the". "get the" string is contained in both placeholders.
    mission_space = MissionSpace(
        mission_func=lambda get_syntax, obj_type: f"{get_syntax} {obj_type}.",
        ordered_placeholders=[
            ["go get the", "get the", "go fetch the", "fetch the"],
            ["ball", "key"],
        ],
    )

    assert mission_space.contains("get the ball.")
    assert mission_space.contains("go get the key.")
    assert mission_space.contains("go fetch the ball.")

    # Test repeated placeholders
    mission_space = MissionSpace(
        mission_func=lambda get_syntax, color_1, obj_type_1, color_2, obj_type_2: f"{get_syntax} {color_1} {obj_type_1} and the {color_2} {obj_type_2}.",
        ordered_placeholders=[
            ["go get the", "get the", "go fetch the", "fetch the"],
            ["green", "red"],
            ["ball", "key"],
            ["green", "red"],
            ["ball", "key"],
        ],
    )

    assert mission_space.contains("get the green key and the green key.")
    assert mission_space.contains("go fetch the red ball and the green key.")


# not reasonable to test for all environments, test for a few of them.
@pytest.mark.parametrize(
    "env_id",
    [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-DoorKey-16x16-v0",
        "MiniGrid-ObstructedMaze-1Dl-v0",
    ],
)
def test_env_sync_vectorization(env_id):
    def env_maker(env_id, **kwargs):
        def env_func():
            env = gym.make(env_id, **kwargs)
            return env

        return env_func

    num_envs = 4
    env = gym.vector.SyncVectorEnv([env_maker(env_id) for _ in range(num_envs)])
    env.reset()
    env.step(env.action_space.sample())
    env.close()
