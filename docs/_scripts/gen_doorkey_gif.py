"""Train a fully observed DoorKey policy and record it for the documentation.

Run from the repository root with ``python docs/_scripts/gen_doorkey_gif.py``.
Training uses tabular Q-learning with the environment's sparse reward; it takes
about two minutes on a CPU. No pretrained model or extra RL package is needed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import minigrid
from minigrid.wrappers import FullyObsWrapper

gym.register_envs(minigrid)

ENV_ID = "MiniGrid-DoorKey-5x5-v0"
OUTPUT = Path(__file__).resolve().parents[1] / "_static/videos/minigrid/DoorKeyEnv.gif"


def train_policy(episodes=10_000, seed=73):
    """Learn action values from full-grid observations and unmodified rewards."""
    rng = np.random.default_rng(seed)
    env = FullyObsWrapper(gym.make(ENV_ID))
    env.reset(seed=seed)
    values = defaultdict(lambda: np.zeros(env.action_space.n))

    for episode in range(episodes):
        obs, _ = env.reset()
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.8))
        while True:
            action_values = values[obs["image"].tobytes()]
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                best = np.flatnonzero(action_values == action_values.max())
                action = int(rng.choice(best))

            obs, reward, terminated, truncated, _ = env.step(action)
            # A time limit does not make the next state terminal for Q-learning.
            target = reward
            if not terminated:
                target += 0.99 * values[obs["image"].tobytes()].max()
            action_values[action] += 0.1 * (target - action_values[action])
            if terminated or truncated:
                break

        if (episode + 1) % 1000 == 0:
            print(f"Trained {episode + 1}/{episodes} episodes", flush=True)

    env.close()
    return dict(values)


def record_policy(values, output=OUTPUT, seed=123):
    """Record a greedy rollout, including the frame after reaching the goal."""
    env = FullyObsWrapper(
        gym.make(ENV_ID, render_mode="rgb_array", tile_size=64, highlight=False)
    )
    frames = []
    try:
        obs, _ = env.reset(seed=seed)
        frames.append(Image.fromarray(env.render()))
        while True:
            action = int(values[obs["image"].tobytes()].argmax())
            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break

        if reward <= 0:
            raise RuntimeError("The learned policy did not reach the goal")

        # Pause at the beginning and end so the task and its outcome are visible.
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=[700] + [300] * (len(frames) - 2) + [1600],
            loop=0,
        )
        print(f"Saved {output} ({len(frames) - 1} steps, reward {reward:.4f})")
    finally:
        env.close()


if __name__ == "__main__":
    record_policy(train_policy())
