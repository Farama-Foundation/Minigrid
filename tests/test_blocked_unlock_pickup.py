from __future__ import annotations

import gymnasium as gym
import pytest

from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.world_object import Box

# Each of these environments places a single empty Box as the object the
# agent must pick up, via RoomGrid.add_object(..., kind="box").
ENVS_WITH_EMPTY_TARGET_BOX = [
    "MiniGrid-UnlockPickup-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
    "BabyAI-UnlockPickup-v0",
    "BabyAI-BlockedUnlockPickup-v0",
]


def find_the_box(unwrapped):
    boxes = [obj for obj in unwrapped.grid.grid if isinstance(obj, Box)]
    assert len(boxes) == 1, f"expected exactly one box, found {len(boxes)}"
    return boxes[0]


@pytest.mark.parametrize("env_id", ENVS_WITH_EMPTY_TARGET_BOX)
def test_toggling_the_target_box_does_not_delete_it(env_id):
    """
    Toggling an empty Box used to replace it with its contents even when it
    had none, which deletes the box from the grid and leaves the episode
    unsolvable.
    """
    env = gym.make(env_id)
    env.reset(seed=0)
    unwrapped = env.unwrapped

    target = find_the_box(unwrapped)
    assert target.contains is None
    box_pos = target.cur_pos

    # Stand the agent on a free tile adjacent to the box, facing it.
    for direction, vec in enumerate(DIR_TO_VEC):
        adjacent = tuple(box_pos - vec)
        if unwrapped.grid.get(*adjacent) is None:
            unwrapped.agent_pos = adjacent
            unwrapped.agent_dir = direction
            break
    else:
        raise AssertionError("no free tile adjacent to the target box")

    env.step(Actions.toggle)

    assert unwrapped.grid.get(*box_pos) is target
