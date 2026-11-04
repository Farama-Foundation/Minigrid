"""Capture additional records by executing the original transition methods."""

from __future__ import annotations

import importlib
import json

import numpy as np
from runtime import SOURCE_COMMIT, load_reference


def states(env):
    return [
        {
            "position": None if a.pos is None else list(map(int, a.pos)),
            "direction": int(a.dir),
            "done": bool(a.done),
            "prestige": float(a.prestige),
            "bonus_state": a.bonus_state,
            "bonuses": [[int(i), float(p)] for i, p in a.bonuses],
        }
        for a in env.agents
    ]


def generate():
    base, agents, objects, empty, _ = load_reference()
    cycle = importlib.import_module("_marlgrid_source.envs.goalcycle")
    visibility = importlib.import_module("_marlgrid_source.envs.viz_test")
    cases = []
    specs = [
        (
            "cycle",
            {"initial_reward": initial, "reset_on_mistake": mistake, "penalty": 0.25},
            [0, 0],
        )
        for initial in (False, True)
        for mistake in (False, True)
    ] + [
        ("cycle", {"reward_decay": True, "penalty": 0.25}, [0, 0]),
        ("cycle_map", {"n_clutter": 6}, [0, 0]),
        ("empty", {"respawn": True}, [0, 0]),
        ("empty", {"respawn": True, "ghost_mode": False}, [0, 0]),
        ("empty", {}, [0, 2]),
        ("empty", {}, [2, 3]),
        ("visibility", {}, [0, 0]),
    ]
    for index, (family, options, delays) in enumerate(specs):
        kwargs = {"max_steps": 12, **options}
        factory = {
            "cycle": cycle.ClutteredGoalCycleEnv,
            "cycle_map": cycle.ClutteredGoalCycleEnv,
            "empty": empty.EmptyMultiGrid,
            "visibility": visibility.VisibilityTestEnv,
        }[family]
        if family.startswith("cycle"):
            kwargs.setdefault("n_clutter", 0)
        env = factory(
            agents=[
                agents.GridAgentInterface(
                    color=c, see_through_walls=True, spawn_delay=d
                )
                for c, d in zip(("red", "blue"), delays)
            ],
            grid_size=9,
            **kwargs,
        )
        env.seed(42)
        env.reset()
        placement = None
        if family == "cycle":
            env.grid = base.MultiGrid((9, 9))
            env.grid.wall_rect(0, 0, 9, 9)
            for bonus_id in range(3):
                env.put_obj(
                    objects.BonusTile(
                        reward=1,
                        penalty=kwargs["penalty"],
                        bonus_id=bonus_id,
                        n_bonus=3,
                        initial_reward=kwargs.get("initial_reward", True),
                        reset_on_mistake=kwargs.get("reset_on_mistake", False),
                    ),
                    2 + bonus_id,
                    3,
                )
            placement = [[1, 3, 0], [1, 4, 0]]
            for a, (x, y, direction) in zip(env.agents, placement):
                a.agents = []
                a.dir = direction
                assert env.try_place_obj(a, (x, y))
            actions = [
                [2, 6],
                [2, 6],
                [2, 6],
                [1, 6],
                [1, 6],
                [2, 6],
                [2, 6],
                [0, 6],
                [0, 6],
                [2, 6],
                [2, 6],
                [6, 6],
            ]
        elif family == "empty" and kwargs.get("respawn"):
            env.grid = base.MultiGrid((9, 9))
            env.grid.wall_rect(0, 0, 9, 9)
            env.put_obj(objects.Goal(color="green", reward=1), 7, 7)
            placement = [[6, 7, 0], [7, 6, 1]]
            for a, (x, y, direction) in zip(env.agents, placement):
                a.agents = []
                a.dir = direction
                assert env.try_place_obj(a, (x, y))
            actions = [[2, 2]] + [[0, 1], [2, 2], [1, 0], [2, 2]] * 3
        else:
            actions = np.random.default_rng(index).integers(0, 3, (12, 2)).tolist()
        cells = []
        for x in range(9):
            for y in range(9):
                obj = env.grid.get(x, y)
                if obj is not None and not obj.is_agent:
                    cells.append([x, y, obj.type, getattr(obj, "bonus_id", None)])
        initial = states(env)
        rows = []
        for joint in actions:
            _, reward, done, _ = env.step(joint)
            rows.append(
                {
                    "actions": joint,
                    "states": states(env),
                    "rewards": reward.tolist(),
                    "done": bool(done),
                }
            )
            if done:
                break
        cases.append(
            {
                "name": f"{family}_{index}",
                "family": family,
                "kwargs": {"size": 9, "num_agents": 2, "spawn_delay": delays, **kwargs},
                "seed": 42,
                "placement": placement,
                "cells": cells,
                "initial": initial,
                "steps": rows,
            }
        )
    return {
        "source_commit": SOURCE_COMMIT,
        "scope": "Original transitions with dependency adapters and see_through_walls=True; excludes image/seed identity.",
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(generate()))
