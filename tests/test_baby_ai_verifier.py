from __future__ import annotations

import gymnasium as gym
import pytest

from minigrid.core.constants import DIR_TO_VEC
from minigrid.envs.babyai.core import verifier
from minigrid.envs.babyai.core.verifier import (
    AfterInstr,
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    SeqInstr,
)


def open_door(env, door, done_action=False):
    """
    Put the agent in front of the door, facing it, and toggle the door.
    The done action is taken afterwards if requested.
    """
    base_env = env.unwrapped
    for direction, vec in enumerate(DIR_TO_VEC):
        pos = (door.cur_pos[0] - vec[0], door.cur_pos[1] - vec[1])
        if base_env.grid.get(*pos) is None:
            base_env.agent_pos, base_env.agent_dir = pos, direction
            break

    step = env.step(base_env.actions.toggle)
    if done_action and not step[2]:
        step = env.step(base_env.actions.done)
    return step


@pytest.mark.parametrize("in_order", [True, False])
@pytest.mark.parametrize(
    "env_id, done_actions",
    [
        ("BabyAI-OpenDoorsOrderN2Debug-v0", False),
        ("BabyAI-OpenDoorsOrderN4Debug-v0", False),
        ("BabyAI-OpenDoorsOrderN2-v0", True),
    ],
)
def test_open_doors_order(env_id, done_actions, in_order, monkeypatch):
    """
    Opening the two doors in the order of the mission should succeed and
    opening the other door first should fail, in the strict (debug) levels
    as well as with done actions.
    """
    monkeypatch.setattr(verifier, "use_done_actions", done_actions)
    env = gym.make(env_id)

    seq_instrs = set()
    for seed in range(10):
        env.reset(seed=seed)

        instrs = env.unwrapped.instrs
        if not isinstance(instrs, SeqInstr):
            continue
        seq_instrs.add(type(instrs))

        first, second = instrs.instr_a, instrs.instr_b
        if isinstance(instrs, AfterInstr):
            first, second = second, first
        first_door, second_door = first.desc.obj_set[0], second.desc.obj_set[0]

        if in_order:
            _, reward, terminated, _, _ = open_door(env, first_door, done_actions)
            assert reward == 0 and not terminated

            _, reward, terminated, _, _ = open_door(env, second_door, done_actions)
            assert reward > 0 and terminated
        else:
            _, reward, terminated, _, _ = open_door(env, second_door, done_actions)
            assert reward == 0 and terminated

    # make sure that both kinds of sequences were tested
    assert seq_instrs == {BeforeInstr, AfterInstr}

    env.close()


@pytest.mark.parametrize("seq_instr", [BeforeInstr, AfterInstr])
def test_action_completes_both_instrs(seq_instr):
    """
    The action that completes the first instruction of a sequence
    can complete the second instruction as well.
    """
    env = gym.make("BabyAI-OpenDoorsOrderN2-v0")
    env.reset(seed=0)

    base_env = env.unwrapped
    door = next(door for door in base_env.get_room(1, 1).doors if door)
    open_instr = OpenInstr(ObjDesc(door.type, door.color))
    go_to_instr = GoToInstr(ObjDesc(door.type, door.color))

    # "open the door, then go to the door" and
    # "go to the door after you open the door"
    if seq_instr is BeforeInstr:
        base_env.instrs = BeforeInstr(open_instr, go_to_instr)
    else:
        base_env.instrs = AfterInstr(go_to_instr, open_instr)
    base_env.instrs.reset_verifier(base_env)

    _, reward, terminated, _, _ = open_door(env, door)
    assert reward > 0 and terminated

    env.close()
