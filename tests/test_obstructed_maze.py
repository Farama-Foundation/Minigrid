from __future__ import annotations

import gymnasium as gym

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Ball, Box


def find_ball_room(env):
    for obj in env.grid.grid:
        if isinstance(obj, Ball) and obj.color == COLOR_NAMES[0]:
            return env.room_from_pos(*obj.cur_pos)


def find_target_key(env, color):
    for obj in env.grid.grid:
        if isinstance(obj, Box) and obj.contains.color == color:
            return True
    return False


def test_env(env_id, repeats=10000):
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


def main():
    """
    Test the frequency of unsolvable situation in this environment, including
    MiniGrid-ObstructedMaze-2Dlhb, -1Q, and -2Q. The reason for the unsolvable
    situation is that in the v0 version of these environments, the box containing
    the key to the door connecting the upper-right room may be covered by the
    blocking ball of the door connecting the lower-right room.

    Covering may also occur in MiniGrid-ObstructedMaze-Full, but it will not
    lead to an unsolvable situation.
    """

    envs_v0 = [
        "MiniGrid-ObstructedMaze-2Dlhb-v0",  # expected: 1 / 15 = 6.67%
        "MiniGrid-ObstructedMaze-1Q-v0",  # expected: 1 / 15 = 6.67%
        "MiniGrid-ObstructedMaze-2Q-v0",  # expected: 1 / 30 = 3.33%
        "MiniGrid-ObstructedMaze-Full-v0",
    ]  # expected: 0
    envs_v1 = [
        "MiniGrid-ObstructedMaze-2Dlhb-v1",
        "MiniGrid-ObstructedMaze-1Q-v1",
        "MiniGrid-ObstructedMaze-2Q-v1",
        "MiniGrid-ObstructedMaze-Full-v1",
    ]

    for env_id in envs_v0:
        print(f"{env_id}: {test_env(env_id):.2f}% unsolvable.")
    for env_id in envs_v1:
        print(f"{env_id}: {test_env(env_id):.2f}% unsolvable.")


if __name__ == "__main__":
    main()
