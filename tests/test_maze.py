from __future__ import annotations

from collections import deque

import pytest

from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.envs.maze import MazeEnv


@pytest.mark.parametrize("size", [5, 9, 11, 15, 31])
@pytest.mark.parametrize("obstacle_type", [Wall, Lava])
def test_perfect_maze_and_solution(size, obstacle_type):
    env = MazeEnv(size=size, obstacle_type=obstacle_type)
    for seed in range(50):
        env.reset(seed=seed)
        # Check the actual tile graph, rather than the generator's room bookkeeping.
        passages = {
            (x, y)
            for x in range(size)
            for y in range(size)
            if env.grid.get(x, y) is None or isinstance(env.grid.get(x, y), Goal)
        }
        for x in range(size):
            assert isinstance(env.grid.get(x, 0), Wall)
            assert isinstance(env.grid.get(x, size - 1), Wall)
            assert isinstance(env.grid.get(0, x), Wall)
            assert isinstance(env.grid.get(size - 1, x), Wall)
        for x in range(1, size - 1):
            for y in range(1, size - 1):
                if (x, y) not in passages:
                    assert isinstance(env.grid.get(x, y), obstacle_type)

        start = tuple(env.agent_pos)
        goal = next(pos for pos in passages if isinstance(env.grid.get(*pos), Goal))
        assert start[0] == 1 and goal[0] == size - 2
        parents = {start: None}
        queue = deque([start])
        edges = 0
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                neighbor = (x + dx, y + dy)
                if neighbor in passages:
                    edges += 1
                    if neighbor not in parents:
                        parents[neighbor] = (x, y)
                        queue.append(neighbor)
        assert set(parents) == passages  # Every passage is connected.
        assert edges // 2 == len(passages) - 1  # No cycles: a perfect maze.
        path = [goal]
        while path[-1] != start:
            path.append(parents[path[-1]])
        path.reverse()
        for current, target in zip(path, path[1:]):
            delta = (target[0] - current[0], target[1] - current[1])
            direction = [tuple(vector) for vector in DIR_TO_VEC].index(delta)
            while env.agent_dir != direction:
                _, reward, terminated, truncated, _ = env.step(env.actions.right)
                assert not terminated and not truncated
                assert reward == 0
            _, reward, terminated, truncated, _ = env.step(env.actions.forward)
            assert tuple(env.agent_pos) == target
            assert not truncated
            assert terminated == (target == goal)
        assert reward == pytest.approx(1 - 0.9 * env.step_count / env.max_steps)
        assert reward > 0
    env.close()


@pytest.mark.parametrize("size", [0, 3, 4, 8, 9.0, True])
def test_invalid_size(size):
    with pytest.raises(ValueError, match="odd integer"):
        MazeEnv(size=size)


def test_invalid_obstacle():
    with pytest.raises(ValueError, match="obstacle_type must be"):
        MazeEnv(obstacle_type=Goal)
