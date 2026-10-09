from __future__ import annotations

import gymnasium as gym
import pytest

from minigrid.core.actions import Actions
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.world_object import Box

# Each of these environments places a single empty Box as the object the
# agent must pick up, via RoomGrid.add_object(..., kind="box"). Their -v0
# registration keeps the original (buggy) dynamics for reproducibility; the
# -v1 registration carries the fix.
DETERMINISTIC_EMPTY_BOX_ENVS = [
    "MiniGrid-UnlockPickup",
    "MiniGrid-BlockedUnlockPickup",
    "BabyAI-UnlockPickup",
    "BabyAI-BlockedUnlockPickup",
]

# Every other registered env family where an empty Box can be drawn at random
# as the pickup goal or the PutNear/PutNext move object, via RoomGrid.add_object()
# or add_distractors() with no fixed kind, or PutNearEnv's own object placement.
# Only some seeds place a box, so these are checked on their -v1 registration
# only, across a handful of seeds.
PROBABILISTIC_EMPTY_BOX_ENVS = [
    "BabyAI-FindObjS5",
    "BabyAI-FindObjS6",
    "BabyAI-FindObjS7",
    "BabyAI-Pickup",
    "BabyAI-UnblockPickup",
    "BabyAI-PickupDist",
    "BabyAI-PickupDistDebug",
    "BabyAI-PickupAbove",
    "BabyAI-PickupLoc",
    "BabyAI-ActionObjDoor",
    "BabyAI-MoveTwoAcrossS5N2",
    "BabyAI-MoveTwoAcrossS8N9",
    "BabyAI-PutNextLocal",
    "BabyAI-PutNextLocalS5N3",
    "BabyAI-PutNextLocalS6N4",
    "BabyAI-PutNextS4N1",
    "BabyAI-PutNextS5N2",
    "BabyAI-PutNextS5N1",
    "BabyAI-PutNextS6N3",
    "BabyAI-PutNextS7N4",
    "BabyAI-Synth",
    "BabyAI-SynthS5R2",
    "BabyAI-SynthLoc",
    "BabyAI-SynthSeq",
    "BabyAI-MiniBossLevel",
    "BabyAI-BossLevel",
    "BabyAI-BossLevelNoUnlock",
    "MiniGrid-PutNear-6x6-N2",
    "MiniGrid-PutNear-8x8-N3",
]

# The rarest of these families draws an empty box on roughly 1 seed in 3;
# 20 seeds makes a miss very unlikely.
SEEDS_PER_PROBABILISTIC_ENV = 20


def find_the_box(unwrapped):
    boxes = [obj for obj in unwrapped.grid.grid if isinstance(obj, Box)]
    assert len(boxes) == 1, f"expected exactly one box, found {len(boxes)}"
    return boxes[0]


def find_empty_boxes(unwrapped):
    return [
        obj
        for obj in unwrapped.grid.grid
        if isinstance(obj, Box) and obj.contains is None
    ]


def face_object(unwrapped, obj):
    """Move the agent to a free tile adjacent to obj and face it.

    Returns whether such a tile was found.
    """
    pos = obj.cur_pos
    for direction, vec in enumerate(DIR_TO_VEC):
        adjacent = tuple(pos - vec)
        if unwrapped.grid.get(*adjacent) is None:
            unwrapped.agent_pos = adjacent
            unwrapped.agent_dir = direction
            return True
    return False


@pytest.mark.parametrize("env_id", DETERMINISTIC_EMPTY_BOX_ENVS)
def test_toggling_the_target_box_deletes_it_on_v0(env_id):
    """
    Toggling an empty Box on the original -v0 registration replaces it with
    its contents even when it has none, which deletes the box from the grid
    and leaves the episode unsolvable.
    """
    env = gym.make(f"{env_id}-v0")
    env.reset(seed=0)
    unwrapped = env.unwrapped

    target = find_the_box(unwrapped)
    assert target.contains is None
    box_pos = target.cur_pos
    assert face_object(unwrapped, target), "no free tile adjacent to the target box"

    env.step(Actions.toggle)

    assert unwrapped.grid.get(*box_pos) is None


@pytest.mark.parametrize("env_id", DETERMINISTIC_EMPTY_BOX_ENVS)
def test_toggling_the_target_box_survives_on_v1(env_id):
    """
    The -v1 registration fixes toggling an empty Box so it no longer deletes
    the box, keeping the episode solvable.
    """
    env = gym.make(f"{env_id}-v1")
    env.reset(seed=0)
    unwrapped = env.unwrapped

    target = find_the_box(unwrapped)
    assert target.contains is None
    box_pos = target.cur_pos
    assert face_object(unwrapped, target), "no free tile adjacent to the target box"

    env.step(Actions.toggle)

    assert unwrapped.grid.get(*box_pos) is target


@pytest.mark.parametrize("env_id", PROBABILISTIC_EMPTY_BOX_ENVS)
def test_v1_never_deletes_an_empty_box(env_id):
    """
    These env families draw the pickup goal or move object at random, so an
    empty Box only appears on some seeds. Wherever one does appear, the -v1
    registration must never delete it on toggle.
    """
    checked_any_box = False
    for seed in range(SEEDS_PER_PROBABILISTIC_ENV):
        env = gym.make(f"{env_id}-v1")
        env.reset(seed=seed)
        unwrapped = env.unwrapped

        for target in find_empty_boxes(unwrapped):
            box_pos = target.cur_pos
            if not face_object(unwrapped, target):
                continue
            checked_any_box = True

            env.step(Actions.toggle)

            assert unwrapped.grid.get(*box_pos) is target

        # Box.toggle only branches on self.contains and fix_empty_box_bug, not
        # on anything seed-dependent, so once a box in this seed has passed,
        # further seeds only add redundant coverage of the same code path.
        if checked_any_box:
            break

    assert (
        checked_any_box
    ), f"no empty Box appeared in {SEEDS_PER_PROBABILISTIC_ENV} seeds for {env_id}"
