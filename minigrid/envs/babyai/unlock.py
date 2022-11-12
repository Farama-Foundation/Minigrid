"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with the `Unlock` instruction.
"""
from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Ball, Box, Key
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, OpenInstr, PickupInstr


class Unlock(RoomGridLevel):
    """
    Unlock a door.

    Competencies: Maze, Open, Unlock. No unblocking.
    """

    def gen_mission(self):
        # Add a locked door to a random room
        id = self._rand_int(0, self.num_cols)
        jd = self._rand_int(0, self.num_rows)
        door, pos = self.add_door(id, jd, locked=True)
        locked_room = self.get_room(id, jd)

        # Add the key to a different room
        while True:
            ik = self._rand_int(0, self.num_cols)
            jk = self._rand_int(0, self.num_rows)
            if ik is id and jk is jd:
                continue
            self.add_object(ik, jk, "key", door.color)
            break

        # With 50% probability, ensure that the locked door is the only
        # door of that color
        if self._rand_bool():
            colors = list(filter(lambda c: c is not door.color, COLOR_NAMES))
            self.connect_all(door_colors=colors)
        else:
            self.connect_all()

        # Add distractors to all but the locked room.
        # We do this to speed up the reachability test,
        # which otherwise will reject all levels with
        # objects in the locked room.
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if i is not id or j is not jd:
                    self.add_distractors(i, j, num_distractors=3, all_unique=False)

        # The agent must be placed after all the object to respect constraints
        while True:
            self.place_agent()
            start_room = self.room_from_pos(*self.agent_pos)
            # Ensure that we are not placing the agent in the locked room
            if start_room is locked_room:
                continue
            break

        self.check_objs_reachable()

        self.instrs = OpenInstr(ObjDesc(door.type, door.color))


class UnlockLocal(RoomGridLevel):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, distractors=False, **kwargs):
        self.distractors = distractors
        super().__init__(**kwargs)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, "key", door.color)
        if self.distractors:
            self.add_distractors(1, 1, num_distractors=3)
        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class KeyInBox(RoomGridLevel):
    """
    Unlock a door. Key is in a box (in the current room).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)

        # Put the key in the box, then place the box in the room
        key = Key(door.color)
        box = Box(self._rand_color(), key)
        self.place_in_room(1, 1, box)

        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class UnlockPickup(RoomGridLevel):
    """
    Unlock a door, then pick up a box in another room
    """

    def __init__(self, distractors=False, max_steps: int | None = None, **kwargs):
        self.distractors = distractors
        room_size = 6
        if max is None:
            max_steps = 8 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=6, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        # Add a random object to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)
        if self.distractors:
            self.add_distractors(num_distractors=4)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class BlockedUnlockPickup(RoomGridLevel):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0] - 1, pos[1], Ball(color))
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class UnlockToUnlock(RoomGridLevel):
    """
    Unlock a door A that requires to unlock a door B before
    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        if max_steps is None:
            max_steps = 30 * room_size**2

        super().__init__(
            num_rows=1, num_cols=3, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=True)

        # Add a key of color A in the room on the right
        self.add_object(2, 0, kind="key", color=colors[0])

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[1], locked=True)

        # Add a key of color B in the middle room
        self.add_object(1, 0, kind="key", color=colors[1])

        obj, _ = self.add_object(0, 0, kind="ball")

        self.place_agent(1, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))
