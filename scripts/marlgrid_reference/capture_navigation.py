"""Print transition fixtures produced by pinned, unmodified marlgrid sources.

Run in a separate process; dependency shims must not leak into the test runner.
Visibility is disabled through the original see_through_walls option. This
comparison deliberately excludes rendering and observation equivalence.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
from runtime import SOURCE, SOURCE_COMMIT, load_reference


def snapshot(env):
    return [
        [int(a.pos[0]), int(a.pos[1]), int(a.dir), bool(a.done)] for a in env.agents
    ]


def generate():
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    assert revision == SOURCE_COMMIT
    assert not subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=SOURCE, text=True
    ).strip()
    base, agents, objects, empty, cluttered = load_reference()
    specifications = [
        ("empty_overlap", "empty", {"ghost_mode": True}, 0, None, None),
        ("empty_exclusive", "empty", {"ghost_mode": False}, 7, None, None),
        ("clutter_density", "cluttered", {"clutter_density": 0.15}, 11, None, None),
        (
            "clutter_random_goal",
            "cluttered",
            {"n_clutter": 6, "randomize_goal": True},
            3,
            None,
            None,
        ),
        (
            "ghost_contention",
            "empty",
            {},
            42,
            [[2, 3, 0], [4, 3, 2]],
            [[2, 2], [2, 6], [0, 2]],
        ),
        (
            "exclusive_contention",
            "empty",
            {"ghost_mode": False},
            42,
            [[2, 3, 0], [4, 3, 2]],
            [[2, 2], [2, 6], [0, 2]],
        ),
        ("ghost_swap", "empty", {}, 7, [[2, 3, 0], [3, 3, 2]], [[2, 2], [2, 2]]),
        (
            "success_and_continuation",
            "empty",
            {},
            42,
            [[6, 7, 0], [7, 5, 1]],
            [[2, 2], [6, 2]],
        ),
        (
            "undecayed_reward",
            "empty",
            {"reward_decay": False},
            42,
            [[6, 7, 0], [7, 5, 1]],
            [[2, 2], [6, 2]],
        ),
        (
            "same_step_success",
            "empty",
            {"ghost_mode": False},
            42,
            [[6, 7, 0], [7, 6, 1]],
            [[2, 2]],
        ),
        (
            "success_at_limit",
            "empty",
            {"max_steps": 1},
            42,
            [[6, 7, 0], [4, 3, 2]],
            [[2, 6]],
        ),
    ]
    cases = []
    for name, family, options, seed, placement, joint_actions in specifications:
        size = 11 if family == "cluttered" else 9
        count = 3 if family == "cluttered" else 2
        kwargs = {"max_steps": 12, **options}
        env_class = (
            empty.EmptyMultiGrid if family == "empty" else cluttered.ClutteredMultiGrid
        )
        env = env_class(
            agents=[
                agents.GridAgentInterface(color=c, see_through_walls=True)
                for c in ["red", "blue", "green"][:count]
            ],
            grid_size=size,
            **kwargs,
        )
        env.seed(seed)
        env.reset()
        walls = [
            [x, y]
            for x in range(size)
            for y in range(size)
            if isinstance(env.grid.get(x, y), objects.Wall)
        ]
        goal = next(
            [x, y]
            for x in range(size)
            for y in range(size)
            if isinstance(env.grid.get(x, y), objects.Goal)
        )
        if placement is not None:
            env.grid = base.MultiGrid((size, size))
            for x, y in walls:
                env.grid.set(x, y, objects.Wall())
            env.put_obj(objects.Goal(color="green", reward=1), *goal)
            for agent, (x, y, direction) in zip(env.agents, placement):
                agent.agents = []
                agent.dir = direction
                assert env.try_place_obj(agent, (x, y))
        initial = snapshot(env)
        if joint_actions is None:
            rng = np.random.default_rng(918 + seed)
            joint_actions = rng.integers(0, 3, size=(12, count)).tolist()
        steps = []
        for action in joint_actions:
            _, rewards, done, _ = env.step(action)
            steps.append(
                {
                    "actions": action,
                    "states": snapshot(env),
                    "rewards": rewards.tolist(),
                    "done": bool(done),
                }
            )
            if done:
                break
        cases.append(
            {
                "name": name,
                "family": family,
                "kwargs": {"size": size, "num_agents": count, **kwargs},
                "seed": seed,
                "placement": placement,
                "walls": walls,
                "goal": goal,
                "initial": initial,
                "steps": steps,
            }
        )
    return {
        "source": {
            "repository": "https://github.com/kandouss/marlgrid",
            "commit": SOURCE_COMMIT,
            "scope": "Map, placement, direction, reward and termination; not observation equivalence.",
            "rng": "Gym RNG adapter uses numpy.default_rng and aliases randint to integers; not historical Gym seed identity.",
            "compatibility": "Gymnasium spaces, NumPy legacy aliases, current rendering helpers, no-op numba JIT; original see_through_walls=True.",
        },
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(generate()))
