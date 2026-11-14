from __future__ import annotations

import gymnasium as gym
import pytest

from minigrid.core.roomgrid import RoomGrid


@pytest.mark.parametrize("env_id", ["BabyAI-GoToImpUnlock-v0", "BabyAI-Unlock-v0"])
def test_locked_room_gets_no_key_and_no_distractors(env_id, monkeypatch):
    """The key and the distractors must be placed outside of the locked room.

    The room indices are numpy integers, so comparing them with ``is`` is always
    false and the locked room was not excluded. GoToImpUnlock deliberately puts a
    single object into the locked room at the end, which is the mission target.
    """
    events = []
    add_door, add_object, add_distractors = (
        RoomGrid.add_door,
        RoomGrid.add_object,
        RoomGrid.add_distractors,
    )

    def record_door(self, i, j, door_idx=None, color=None, locked=None):
        if locked:  # only the locked door of the mission, not the ones from connect_all
            events.append(("door", i, j, None))
        return add_door(self, i, j, door_idx, color, locked)

    in_distractors = False

    def record_object(self, i, j, kind=None, *args, **kwargs):
        # Only the mission key counts, not a key that was picked as a random distractor.
        if not in_distractors and kind == "key":
            events.append(("key", i, j, None))
        return add_object(self, i, j, kind, *args, **kwargs)

    def record_distractors(self, i=None, j=None, num_distractors=10, all_unique=True):
        nonlocal in_distractors
        events.append(("distractors", i, j, num_distractors))
        in_distractors = True
        try:
            return add_distractors(self, i, j, num_distractors, all_unique)
        finally:
            in_distractors = False

    monkeypatch.setattr(RoomGrid, "add_door", record_door)
    monkeypatch.setattr(RoomGrid, "add_object", record_object)
    monkeypatch.setattr(RoomGrid, "add_distractors", record_distractors)

    env = gym.make(env_id)
    for seed in range(20):
        env.reset(seed=seed)

    locked_room = None
    for kind, i, j, num_distractors in events:
        if kind == "door":
            locked_room = (i, j)  # a new mission generation attempt starts
        elif kind == "key":
            assert (i, j) != locked_room
        elif kind == "distractors" and num_distractors != 1:
            assert (i, j) != locked_room
