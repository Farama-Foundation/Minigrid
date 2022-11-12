"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission, with different instructions than those in other files.
"""
from __future__ import annotations

from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import (
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)


class ActionObjDoor(RoomGridLevel):
    """
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, **kwargs):
        super().__init__(room_size=7, **kwargs)

    def gen_mission(self):
        objs = self.add_distractors(1, 1, num_distractors=5)
        for _ in range(4):
            door, _ = self.add_door(1, 1, locked=False)
            objs.append(door)

        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)

        if obj.type == "door":
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = OpenInstr(desc)
        else:
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = PickupInstr(desc)


class FindObjS5(RoomGridLevel):
    """
    Pick up an object (in a random room)
    Rooms have a size of 5
    This level requires potentially exhaustive exploration
    """

    def __init__(self, room_size=5, max_steps: int | None = None, **kwargs):

        if max_steps is None:
            max_steps = 20 * room_size**2

        super().__init__(room_size=room_size, max_steps=max_steps, **kwargs)

    def gen_mission(self):
        # Add a random object to a random room
        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(i, j)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class KeyCorridor(RoomGridLevel):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.obj_type = obj_type

        if max_steps is None:
            max_steps = 30 * room_size**2

        super().__init__(
            room_size=room_size, num_rows=num_rows, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), "key", door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class OneRoomS8(RoomGridLevel):
    """
    Pick up the ball
    Rooms have a size of 8
    """

    def __init__(self, room_size=8, **kwargs):
        super().__init__(room_size=room_size, num_rows=1, num_cols=1, **kwargs)

    def gen_mission(self):
        obj, _ = self.add_object(0, 0, kind="ball")
        self.place_agent()
        self.instrs = PickupInstr(ObjDesc(obj.type))


class MoveTwoAcross(RoomGridLevel):
    """
    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.
    """

    def __init__(
        self, room_size, objs_per_room, max_steps: int | None = None, **kwargs
    ):
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room

        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        objs_l = self._rand_subset(objs_l, 2)
        objs_r = self._rand_subset(objs_r, 2)
        a = objs_l[0]
        b = objs_r[0]
        c = objs_r[1]
        d = objs_l[1]

        self.instrs = BeforeInstr(
            PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color)),
            PutNextInstr(ObjDesc(c.type, c.color), ObjDesc(d.type, d.color)),
        )
