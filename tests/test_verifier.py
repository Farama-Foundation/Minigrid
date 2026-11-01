from __future__ import annotations

import pytest

from minigrid.core.world_object import Ball, Key
from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PutNextInstr


class PutNextTestEnv(RoomGridLevel):
    def __init__(self, strict):
        self.strict = strict
        super().__init__(room_size=7, num_rows=1, num_cols=1, max_steps=30)

    def gen_mission(self):
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.moving_object = Ball("red")
        self.fixed_object = Key("blue")
        self.distractor = Ball("green")
        self.put_obj(self.moving_object, 2, 1)
        self.put_obj(self.fixed_object, 3, 3)
        self.put_obj(self.distractor, 1, 2)
        # The first matching ball in grid order is not the one picked up.
        self.put_obj(Ball("red"), 1, 4)
        self.instrs = PutNextInstr(
            ObjDesc("ball", "red"), ObjDesc("key", "blue"), strict=self.strict
        )


@pytest.mark.parametrize("strict", [False, True])
def test_putnext_correct_pickup_can_complete_mission(strict):
    env = PutNextTestEnv(strict)
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, _ = env.step(env.actions.pickup)
        assert env.carrying is env.moving_object
        assert reward == 0
        assert not terminated and not truncated

        for action in (env.actions.forward, env.actions.forward, env.actions.right):
            _, reward, terminated, truncated, _ = env.step(action)
            assert reward == 0
            assert not terminated and not truncated

        _, reward, terminated, truncated, _ = env.step(env.actions.drop)
        assert env.carrying is None
        assert env.grid.get(3, 2) is env.moving_object
        assert terminated and not truncated
        assert reward == env._reward()
    finally:
        env.close()


@pytest.mark.parametrize("strict", [False, True])
def test_putnext_wrong_pickup_only_fails_in_strict_mode(strict):
    env = PutNextTestEnv(strict)
    try:
        env.reset(seed=0)
        env.step(env.actions.right)
        _, reward, terminated, truncated, _ = env.step(env.actions.pickup)
        assert env.carrying is env.distractor
        assert reward == 0
        assert terminated == strict
        assert not truncated
    finally:
        env.close()


def test_putnext_unsuccessful_pickup_does_not_fail():
    env = PutNextTestEnv(strict=True)
    try:
        env.reset(seed=0)
        env.step(env.actions.left)
        _, reward, terminated, truncated, _ = env.step(env.actions.pickup)
        assert env.carrying is None
        assert reward == 0
        assert not terminated and not truncated
    finally:
        env.close()
