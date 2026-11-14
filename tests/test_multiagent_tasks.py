from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import data_equivalence

from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Floor, Goal, Key, Lava, Wall
from minigrid.multiagent import (
    ClutteredParallelEnv,
    DoorKeyParallelEnv,
    EmptyParallelEnv,
    GoalCycleParallelEnv,
    MarlgridListAdapter,
    VisibilityParallelEnv,
    make_marlgrid,
)
from minigrid.multiagent.tasks import _BonusTile

pz_tests = pytest.importorskip("pettingzoo.test")


@pytest.mark.parametrize(
    "env_class", [DoorKeyParallelEnv, GoalCycleParallelEnv, VisibilityParallelEnv]
)
def test_new_task_apis(env_class):
    from pettingzoo.utils import parallel_to_aec

    env = env_class(max_steps=20)
    pz_tests.parallel_api_test(env, num_cycles=50)
    env.close()
    pz_tests.parallel_seed_test(lambda: env_class(max_steps=20), num_cycles=50)
    converted = parallel_to_aec(env_class(max_steps=20))
    pz_tests.api_test(converted, num_cycles=50)
    converted.close()


@pytest.mark.parametrize(
    "options",
    [
        {"respawn": True},
        {"spawn_delay": [0, 3]},
        {"spawn_delay": 3},
        {"spawn_delay": 100},
        {"respawn": True, "spawn_delay": [2, 0]},
        {"restrict_actions": True},
        {"observation_style": "image"},
        {
            "observation_style": "rich",
            "observe_rewards": True,
            "observe_position": True,
            "observe_orientation": True,
        },
        {"view_offset": 1},
    ],
)
def test_optional_modes_parallel_api_and_seed(options):
    env = EmptyParallelEnv(max_steps=10, **options)
    pz_tests.parallel_api_test(env, num_cycles=30)
    pz_tests.parallel_seed_test(
        lambda: EmptyParallelEnv(max_steps=10, **options), num_cycles=30
    )
    env.close()


def place(env, positions):
    for agent, (x, y, direction) in zip(env.possible_agents, positions):
        env.agent_states[agent].position = (x, y)
        env.agent_states[agent].direction = direction
    env._visible_agents = env.agents[:]


def test_doorkey_shared_inventory_and_complete_rollout():
    env = DoorKeyParallelEnv(size=7, ghost_mode=False)
    env.reset(seed=1)
    env.grid = Grid(7, 7)
    env.grid.wall_rect(0, 0, 7, 7)
    env.grid.vert_wall(3, 0)
    env.grid.set(3, 3, Door("yellow", is_locked=True))
    env.grid.set(2, 3, Key("yellow"))
    env.grid.set(5, 3, Goal())
    place(env, [(1, 3, 0), (2, 2, 1)])
    # Contention: only one agent may take the shared key.
    env.step({a: 3 for a in env.agents})
    carrier = next(a for a, s in env.agent_states.items() if s.carrying is not None)
    peer = next(a for a in env.agents if a != carrier)
    assert sum(s.carrying is not None for s in env.agent_states.values()) == 1
    env.agent_states[carrier].position, env.agent_states[carrier].direction = (2, 3), 0
    env.agent_states[peer].position, env.agent_states[peer].direction = (2, 2), 1
    env.step({carrier: 6, peer: 5})
    assert env.grid.get(3, 3).is_locked
    env.step({carrier: 5, peer: 6})
    assert env.grid.get(3, 3).is_open
    for _ in range(3):
        _, rewards, term, _, _ = env.step({carrier: 2, peer: 6})
    assert term[carrier] and rewards[carrier] > 0
    assert env.agents == [peer]
    env.agent_states[peer].position, env.agent_states[peer].direction = (2, 3), 0
    for _ in range(3):
        _, rewards, term, _, _ = env.step({peer: 2})
    assert term[peer] and rewards[peer] > 0 and not env.agents


def test_doorkey_wrong_key_does_not_open_door():
    env = DoorKeyParallelEnv()
    env.reset(seed=3)
    env.agent_states["agent_0"].position = (env.door_pos[0] - 1, env.door_pos[1])
    env.agent_states["agent_0"].direction = 0
    env.agent_states["agent_0"].carrying = Key("red")
    env.step({"agent_0": 5, "agent_1": 6})
    assert env.grid.get(*env.door_pos).is_locked


@pytest.mark.parametrize("left_only", [False, True])
def test_doorkey_layout_seed_and_spawn_region(left_only):
    env = DoorKeyParallelEnv(start_in_left_room=left_only)
    seen_right = False
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        states = copy.deepcopy(env.agent_states)
        grid = env.grid.encode()
        assert env.grid.get(*env.door_pos).is_locked
        keys = [
            (x, y)
            for x in range(env.size)
            for y in range(env.size)
            if isinstance(env.grid.get(x, y), Key)
        ]
        assert len(keys) == 1 and keys[0][0] < env.split_index
        for state in states.values():
            seen_right |= state.position[0] > env.split_index
            if left_only:
                assert state.position[0] < env.split_index
        again, _ = env.reset(seed=seed)
        assert data_equivalence(obs, again) and env.agent_states == states
        np.testing.assert_array_equal(grid, env.grid.encode())
    assert seen_right != left_only


def cycle_env(**kwargs):
    env = GoalCycleParallelEnv(size=7, n_clutter=0, n_bonus_tiles=3, **kwargs)
    env.reset(seed=3)
    # Keep the generated bonus objects, arranging a deterministic visitation path.
    tiles = [env.grid.get(*p) for p in env.bonus_positions]
    env.grid = Grid(7, 7)
    env.grid.wall_rect(0, 0, 7, 7)
    for i, tile in enumerate(tiles):
        env.grid.set(2 + i, 3, tile)
    place(env, [(1, 3, 0), (1, 4, 0)])
    return env


def enter_bonus(env, agent, index):
    state = env.agent_states[agent]
    state.position, state.direction = (1 + index, 3), 0
    return env.step({a: 2 if a == agent else 6 for a in env.agents})


@pytest.mark.parametrize("initial_reward", [False, True])
@pytest.mark.parametrize("reset_on_mistake", [False, True])
def test_cycle_sequence_mistakes_and_independent_state(
    initial_reward, reset_on_mistake
):
    env = cycle_env(
        initial_reward=initial_reward, reset_on_mistake=reset_on_mistake, penalty=0.25
    )
    _, r, term, _, _ = enter_bonus(env, "agent_0", 0)
    assert r["agent_0"] == float(initial_reward) and not any(term.values())
    assert env.agent_states["agent_1"].bonus_state is None
    _, r, _, _, _ = enter_bonus(env, "agent_0", 0)
    assert r["agent_0"] == -0.25
    _, r, _, _, _ = enter_bonus(env, "agent_0", 2)
    assert r["agent_0"] == -0.25
    assert env.agent_states["agent_0"].bonus_state == (2 if reset_on_mistake else 0)
    _, r, _, _, _ = enter_bonus(env, "agent_0", 0 if reset_on_mistake else 1)
    assert r["agent_0"] == 1
    _, r, _, _, _ = enter_bonus(env, "agent_1", 2)
    assert r["agent_1"] == float(initial_reward)
    assert env.agent_states["agent_1"].bonus_state == 2


def test_cycle_wraparound_and_time_limit():
    env = cycle_env(max_steps=4)
    for tile in (0, 1, 2, 0):
        _, rewards, term, trunc, _ = enter_bonus(env, "agent_0", tile)
        assert rewards["agent_0"] == 1 and not any(term.values())
    assert all(trunc.values()) and not env.agents


def test_cycle_single_tile_retains_source_repeat_penalty():
    env = GoalCycleParallelEnv(n_bonus_tiles=1, n_clutter=0, penalty=0.2)
    env.reset(seed=0)
    # The one-tile source regards its sole tile as a repeat, including first entry.
    reward, terminal = env._enter_cell("agent_0", env.grid.get(*env.bonus_positions[0]))
    assert reward == -0.2 and not terminal


def test_cycle_negative_prestige_and_reset():
    env = cycle_env(penalty=2, prestige_beta=0.5, allow_negative_prestige=True)
    enter_bonus(env, "agent_0", 0)
    state = env.agent_states["agent_0"]
    assert state.prestige == 0.5 and state.bonuses == [(0, 1)]
    enter_bonus(env, "agent_0", 0)
    assert state.prestige == -0.75 and state.bonuses[-1] == (0, -1.5)
    env.reset(seed=0)
    assert all(
        s.prestige == 0 and s.bonus_state is None and not s.bonuses
        for s in env.agent_states.values()
    )


def test_cycle_nonnegative_prestige_resets_after_penalty():
    env = cycle_env(penalty=0.1)
    enter_bonus(env, "agent_0", 0)
    enter_bonus(env, "agent_0", 0)
    assert env.agent_states["agent_0"].prestige == 0


def test_delayed_activation_before_increment_and_ignored_actions():
    env = EmptyParallelEnv(spawn_delay=[0, 2])
    obs, info = env.reset(seed=0)
    assert not info["agent_1"]["active"] and not obs["agent_1"]["image"].any()
    for _ in range(2):
        env.step({"agent_0": 6, "agent_1": 1})
        assert env.agent_states["agent_1"].position is None
        assert env.agent_states["agent_1"].direction == 0
    _, _, _, _, info = env.step({"agent_0": 6, "agent_1": 1})
    assert info["agent_1"]["active"] and env.agent_states["agent_1"].direction == 1


def test_delayed_agent_keeps_episode_alive_after_peer_success():
    env = EmptyParallelEnv(spawn_delay=[0, 2])
    env.reset(seed=0)
    env.agent_states["agent_0"].position = (6, 7)
    _, _, term, _, _ = env.step({"agent_0": 2, "agent_1": 6})
    assert term["agent_0"] and env.agents == ["agent_1"]
    env.step({"agent_1": 6})
    env.step({"agent_1": 6})
    assert env.agent_states["agent_1"].position is not None


@pytest.mark.parametrize("terminal_cell", [Goal, Lava])
def test_respawn_is_not_parallel_termination(terminal_cell):
    env = EmptyParallelEnv(respawn=True, max_steps=2)
    env.reset(seed=0)
    place(env, [(2, 3, 0), (4, 4, 0)])
    env.grid.set(3, 3, terminal_cell())
    env.agent_states["agent_0"].carrying = Key("red")
    obs, reward, term, trunc, info = env.step({"agent_0": 2, "agent_1": 6})
    assert not any(term.values()) and not any(trunc.values())
    assert info["agent_0"]["respawned"] and len(env.agents) == 2
    assert env.agent_states["agent_0"].carrying is None
    assert env.observation_space("agent_0").contains(obs["agent_0"])
    assert reward["agent_0"] == (0.55 if terminal_cell is Goal else 0)
    _, _, term, trunc, _ = env.step({a: 6 for a in env.agents})
    assert all(trunc.values()) and not any(term.values())


def test_respawn_preserves_cycle_state_and_direction():
    env = cycle_env(respawn=True)
    enter_bonus(env, "agent_0", 1)
    state = env.agent_states["agent_0"]
    state.position, state.direction = (4, 4), 3
    env.grid.set(4, 3, Lava())
    _, _, _, _, info = env.step({"agent_0": 2, "agent_1": 6})
    assert info["agent_0"]["respawned"]
    assert state.bonus_state == 1 and state.direction == 3 and len(state.bonuses) == 1


@pytest.mark.parametrize("style", ["symbolic", "image", "rich"])
@pytest.mark.parametrize("offset", [0, 1, 6])
def test_observation_spaces_hidden_agents_and_no_mutation(style, offset):
    env = EmptyParallelEnv(
        observation_style=style,
        view_offset=offset,
        observe_rewards=True,
        observe_position=True,
        observe_orientation=True,
        spawn_delay=[0, 5],
    )
    obs, _ = env.reset(seed=3)
    before = env.grid.encode().copy()
    states = copy.deepcopy(env.agent_states)
    for a in env.agents:
        assert env.observation_space(a).contains(obs[a])
        assert data_equivalence(obs[a], env.observe(a))
    assert states == env.agent_states
    np.testing.assert_array_equal(before, env.grid.encode())
    if style == "rich":
        np.testing.assert_array_equal(obs["agent_1"]["pov"], 0)
        assert obs["agent_1"]["reward"] == 0


def test_rich_reward_reflects_current_transition():
    env = EmptyParallelEnv(observation_style="rich", observe_rewards=True)
    env.reset(seed=0)
    env.agent_states["agent_0"].position = (6, 7)
    obs, rewards, _, _, _ = env.step({"agent_0": 2, "agent_1": 6})
    assert obs["agent_0"]["reward"] == pytest.approx(rewards["agent_0"])
    assert not obs["agent_0"]["pov"].any()


def test_hiding_objects_changes_only_observation():
    env = EmptyParallelEnv(hide_item_types=["key"], see_through_walls=True)
    env.reset(seed=0)
    place(env, [(2, 3, 0), (1, 1, 0)])
    env.grid.set(3, 3, Key("red"))
    assert env.observe("agent_0")["image"][3, 5, 0] == 1  # empty
    assert isinstance(env.grid.get(3, 3), Key)
    env.step({"agent_0": 3, "agent_1": 6})
    assert isinstance(env.agent_states["agent_0"].carrying, Key)


@pytest.mark.parametrize(
    "env_class",
    [
        EmptyParallelEnv,
        ClutteredParallelEnv,
        DoorKeyParallelEnv,
        GoalCycleParallelEnv,
        VisibilityParallelEnv,
    ],
)
@pytest.mark.parametrize("style", ["symbolic", "image", "rich"])
def test_list_adapter_matches_native_transitions(env_class, style):
    native = env_class(
        num_agents=2,
        observation_style=style,
        max_steps=12,
        spawn_delay=[0, 2],
        respawn=True,
    )
    adapted = MarlgridListAdapter(pickle.loads(pickle.dumps(native)))
    adapted.seed(42)
    listed = adapted.reset()
    direct, _ = native.reset(seed=42)
    assert data_equivalence(listed, [direct[a] for a in native.possible_agents])
    assert adapted.observation_space.contains(tuple(listed))
    for _ in range(12):
        actions = [native.action_space(a).sample() for a in native.possible_agents]
        direct, rewards, term, trunc, _ = native.step(
            dict(zip(native.possible_agents, actions))
        )
        listed, reward_array, done, info = adapted.step(actions)
        assert data_equivalence(listed, [direct[a] for a in native.possible_agents])
        np.testing.assert_array_equal(
            reward_array, [rewards[a] for a in native.possible_agents]
        )
        assert done == (not native.agents)
        assert info["terminations"] == term and info["truncations"] == trunc
        assert native.agent_states == adapted.parallel_env.agent_states


def test_list_adapter_retains_completed_slots_and_validates_atomically():
    env = EmptyParallelEnv()
    adapter = MarlgridListAdapter(env)
    with pytest.raises(RuntimeError, match="reset"):
        adapter.step([6, 6])
    adapter.reset(seed=0)
    env.agent_states["agent_0"].position = (6, 7)
    _, _, done, _ = adapter.step([2, 6])
    assert not done and env.agents == ["agent_1"]
    for bad in ([6], [9, 6], {"agent_0": 6, "agent_1": 6}, "66"):
        before = copy.deepcopy(env.np_random.bit_generator.state)
        with pytest.raises(ValueError):
            adapter.step(bad)
        assert env.step_count == 1 and env.np_random.bit_generator.state == before
    obs, rewards, _, _ = adapter.step([6, 6])
    assert len(obs) == 2 and rewards[0] == 0 and not obs[0]["image"].any()


def test_multiagent_render_and_import_do_not_change_single_agent_registry_or_tiles():
    baseline = gym.make("MiniGrid-Empty-5x5-v0")
    initial, _ = baseline.reset(seed=42)
    names = set(gym.envs.registry)
    floor = Grid.render_tile(Floor("yellow")).copy()
    env = GoalCycleParallelEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.render()
    assert set(gym.envs.registry) == names
    np.testing.assert_array_equal(floor, Grid.render_tile(Floor("yellow")))
    again, _ = baseline.reset(seed=42)
    assert data_equivalence(initial, again)
    env.close()
    baseline.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spawn_delay": -1},
        {"spawn_delay": True},
        {"spawn_delay": [1]},
        {"spawn_delay": [0, -1]},
        {"prestige_beta": 1.1},
        {"prestige_beta": float("nan")},
        {"respawn": "true"},
        {"allow_negative_prestige": 1},
        {"observation_style": "unknown"},
        {"view_tile_size": 0},
        {"view_offset": 7},
        {"observe_rewards": "false"},
        {"hide_item_types": ["unknown"]},
        {"restrict_actions": "false"},
    ],
)
def test_invalid_new_options(kwargs):
    with pytest.raises(ValueError):
        EmptyParallelEnv(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bonus_tiles": 0},
        {"n_bonus_tiles": 257},
        {"n_bonus_tiles": 1.5},
        {"reward": float("nan")},
        {"penalty": float("inf")},
        {"initial_reward": "false"},
        {"reset_on_mistake": 1},
        {"ensure_reachable": True},
        {"randomize_goal": True},
        {"size": 5, "n_bonus_tiles": 8, "n_clutter": 2},
    ],
)
def test_invalid_cycle_options(kwargs):
    with pytest.raises(ValueError):
        GoalCycleParallelEnv(**kwargs)


@pytest.mark.parametrize(
    "name,size,count",
    [
        ("MarlGrid-1AgentCluttered15x15-v0", 11, 1),
        ("MarlGrid-3AgentCluttered11x11-v0", 11, 3),
        ("MarlGrid-3AgentCluttered15x15-v0", 15, 3),
        ("MarlGrid-2AgentEmpty9x9-v0", 9, 2),
        ("MarlGrid-3AgentEmpty9x9-v0", 9, 3),
        ("MarlGrid-4AgentEmpty9x9-v0", 9, 4),
        ("Goalcycle-demo-solo-v0", 13, 1),
    ],
)
def test_legacy_presets_are_explicit_and_do_not_register(name, size, count):
    registered = set(gym.envs.registry)
    env = make_marlgrid(name, max_steps=1)
    observation = env.reset(seed=0)
    assert len(observation) == count and env.parallel_env.size == size
    assert env.observation_space.contains(tuple(observation))
    assert env.step([0] * count)[2]
    assert set(gym.envs.registry) == registered
    env.close()


def test_preset_unknown_name_and_clutter_override():
    with pytest.raises(ValueError, match="Unknown"):
        make_marlgrid("not-an-environment")
    env = make_marlgrid("MarlGrid-3AgentCluttered11x11-v0", n_clutter=0)
    assert env.parallel_env.n_clutter == 0


_EXTENDED_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "marlgrid_extended.json").read_text(
        encoding="utf-8"
    )
)["cases"]


@pytest.mark.parametrize("case", _EXTENDED_CASES, ids=lambda c: c["name"])
def test_extended_original_source_transitions(case):
    factory = {
        "empty": EmptyParallelEnv,
        "cycle": GoalCycleParallelEnv,
        "cycle_map": GoalCycleParallelEnv,
        "visibility": VisibilityParallelEnv,
    }[case["family"]]
    env = factory(**case["kwargs"])
    env.reset(seed=case["seed"])
    if case["placement"] is not None:
        env.grid = Grid(env.size, env.size)
        for x, y, kind, index in case["cells"]:
            obj = (
                _BonusTile(index)
                if kind == "BonusTile"
                else {"Wall": Wall, "Goal": Goal}[kind]()
            )
            env.grid.set(x, y, obj)
        place(env, case["placement"])
    else:
        cells = []
        for x in range(env.size):
            for y in range(env.size):
                obj = env.grid.get(x, y)
                if obj is not None:
                    cells.append(
                        [
                            x,
                            y,
                            (
                                "BonusTile"
                                if isinstance(obj, _BonusTile)
                                else type(obj).__name__
                            ),
                            getattr(obj, "bonus_id", None),
                        ]
                    )
        assert cells == case["cells"]
    completed = set()

    def check(expected):
        for a, row in zip(env.possible_agents, expected):
            state = env.agent_states[a]
            assert (None if state.position is None else list(state.position)) == row[
                "position"
            ]
            assert state.direction == row["direction"]
            assert (a in completed) == row["done"]
            assert state.prestige == pytest.approx(row["prestige"])
            assert state.bonus_state == row["bonus_state"]
            np.testing.assert_allclose(state.bonuses, row["bonuses"])

    check(case["initial"])
    for row in case["steps"]:
        joint = {
            a: value
            for a, value in zip(env.possible_agents, row["actions"])
            if a in env.agents
        }
        _, rewards, term, _, _ = env.step(joint)
        completed.update(a for a, done in term.items() if done)
        check(row["states"])
        np.testing.assert_allclose(
            [rewards.get(a, 0) for a in env.possible_agents], row["rewards"]
        )
        assert (not env.agents) == row["done"]
    env.close()
