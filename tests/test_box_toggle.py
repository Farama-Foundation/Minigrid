"""Regression tests for https://github.com/Farama-Foundation/Minigrid/issues/447."""

from __future__ import annotations

import types

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.world_object import Box, Key
from minigrid.envs import BlockedUnlockPickupEnv


def _env_with_grid(grid):
    return types.SimpleNamespace(grid=grid, carrying=None)


def test_toggle_empty_box_keeps_box():
    """Toggling an empty box must not delete it (issue #447)."""
    grid = Grid(5, 5)
    box = Box("purple")
    grid.set(2, 2, box)

    toggled = box.toggle(_env_with_grid(grid), (2, 2))

    assert toggled is False
    assert grid.get(2, 2) is box


def test_toggle_box_with_contents_reveals_them():
    """Toggling a box with contents still opens it (BabyAI KeyInBox)."""
    grid = Grid(5, 5)
    box = Box("red", Key("yellow"))
    grid.set(2, 2, box)

    toggled = box.toggle(_env_with_grid(grid), (2, 2))

    assert toggled is True
    contents = grid.get(2, 2)
    assert isinstance(contents, Key)
    assert contents.color == "yellow"


def test_blocked_unlock_pickup_toggle_does_not_delete_goal_box():
    """Env-level regression: the mission box survives a toggle action."""
    env = BlockedUnlockPickupEnv()
    env.reset(seed=1)

    box_pos = next(
        (x, y)
        for x in range(env.grid.width)
        for y in range(env.grid.height)
        if isinstance(env.grid.get(x, y), Box)
    )
    # stand the agent next to the box, facing it
    for dx, dy, direction in ((1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3)):
        x, y = box_pos[0] + dx, box_pos[1] + dy
        if env.grid.get(x, y) is None:
            env.agent_pos = (x, y)
            env.agent_dir = (direction + 2) % 4
            break
    else:  # pragma: no cover - layout always leaves a free side
        raise AssertionError("no free cell next to the box")

    assert isinstance(env.grid.get(*env.front_pos), Box)

    env.step(Actions.toggle)

    assert isinstance(env.grid.get(*box_pos), Box)
    assert env.carrying is None
