from __future__ import annotations

import gymnasium as gym
import pytest

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Ball, Box

TESTING_ENVS = [
    "MiniGrid-ObstructedMaze-2Dlhb",
    "MiniGrid-ObstructedMaze-1Q",
    "MiniGrid-ObstructedMaze-2Q",
    "MiniGrid-ObstructedMaze-Full",
]


def find_ball_room(env):
    for obj in env.grid.grid:
        if isinstance(obj, Ball) and obj.color == COLOR_NAMES[0]:
            return env.room_from_pos(*obj.cur_pos)


def find_target_key(env, color):
    for obj in env.grid.grid:
        if isinstance(obj, Box) and obj.contains.color == color:
            return True
    return False


def env_test(env_id, repeats=10000):
    env = gym.make(env_id)

    cnt = 0
    for _ in range(repeats):
        env.reset()
        ball_room = find_ball_room(env)
        ball_room_doors = list(filter(None, ball_room.doors))
        keys_exit = [find_target_key(env, door.color) for door in ball_room_doors]
        if not any(keys_exit):
            cnt += 1

    return (cnt / repeats) * 100


@pytest.mark.parametrize("env_id", TESTING_ENVS)
def test_solvable_env(env_id):
    assert env_test(env_id + "-v1") == 0, f"{env_id} is unsolvable."


def main():
    """
    Test the frequency of unsolvable situation in this environment, including
    MiniGrid-ObstructedMaze-2Dlhb, -1Q, -2Q, and -Full. The reason for the unsolvable
    situation is that in the v0 version of these environments, the box containing
    the key to the door connecting the upper-right room may be covered by the
    blocking ball of the door connecting the lower-right room.

    Note: Covering that occurs in MiniGrid-ObstructedMaze-Full won't lead to an
    unsolvable situation.

    Expected probability of unsolvable situation:
    - MiniGrid-ObstructedMaze-2Dlhb-v0: 1 / 15 = 6.67%
    - MiniGrid-ObstructedMaze-1Q-v0: 1/ 15 = 6.67%
    - MiniGrid-ObstructedMaze-2Q-v0: 1 / 30 = 3.33%
    - MiniGrid-ObstructedMaze-Full-v0: 0%
    """

    for env_id in TESTING_ENVS:
        print(f"{env_id}: {env_test(env_id + '-v0'):.2f}% unsolvable.")
    for env_id in TESTING_ENVS:
        print(f"{env_id}: {env_test(env_id + '-v1'):.2f}% unsolvable.")


if __name__ == "__main__":
    main()
