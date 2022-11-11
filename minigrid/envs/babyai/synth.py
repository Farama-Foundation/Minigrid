"""
Copied and adapted from https://github.com/mila-iqia/babyai.
Levels described in the Baby AI ICLR 2019 submission.
The instructions are a synthesis of those from `PutNext`, `Open`, `GoTo`, and `Pickup`.
"""

from __future__ import annotations

from minigrid.envs.babyai.core.levelgen import LevelGen


class Synth(LevelGen):
    """
    Union of all instructions from PutNext, Open, Goto and PickUp. The agent
    may need to move objects around. The agent may have to unlock the door,
    but only if it is explicitly referred by the instruction.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open
    """

    def __init__(self, room_size=8, num_rows=3, num_cols=3, num_dists=18, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=num_dists,
            instr_kinds=["action"],
            locations=False,
            unblocking=True,
            implicit_unlock=False,
            **kwargs,
        )


class SynthS5R2(Synth):
    def __init__(self, **kwargs):
        super().__init__(room_size=5, num_rows=2, num_cols=2, num_dists=7, **kwargs)


class SynthLoc(LevelGen):
    """
    Like Synth, but a significant share of object descriptions involves
    location language like in PickUpLoc. No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc
    """

    def __init__(self, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            instr_kinds=["action"],
            locations=True,
            unblocking=True,
            implicit_unlock=False,
            **kwargs,
        )


class SynthSeq(LevelGen):
    """
    Like SynthLoc, but now with multiple commands, combined just like in GoToSeq.
    No implicit unlocking.

    Competencies: Maze, Unblock, Unlock, GoTo, PickUp, PutNext, Open, Loc, Seq
    """

    def __init__(self, **kwargs):
        # We add many distractors to increase the probability
        # of ambiguous locations within the same room
        super().__init__(
            locations=True, unblocking=True, implicit_unlock=False, **kwargs
        )


class MiniBossLevel(LevelGen):
    def __init__(self, **kwargs):
        super().__init__(
            num_cols=2,
            num_rows=2,
            room_size=5,
            num_dists=7,
            locked_room_prob=0.25,
            **kwargs,
        )


class BossLevel(LevelGen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BossLevelNoUnlock(LevelGen):
    def __init__(self, **kwargs):
        super().__init__(locked_room_prob=0, implicit_unlock=False, **kwargs)
