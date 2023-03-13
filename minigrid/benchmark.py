#!/usr/bin/env python3

from __future__ import annotations

import time

import gymnasium as gym

from minigrid.manual_control import ManualControl
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


def benchmark(env_id, num_resets, num_frames):
    env = gym.make(env_id, render_mode="rgb_array")
    # Benchmark env.reset
    t0 = time.time()
    for i in range(num_resets):
        env.reset()
    t1 = time.time()
    dt = t1 - t0
    reset_time = (1000 * dt) / num_resets

    # Benchmark rendering
    t0 = time.time()
    for i in range(num_frames):
        env.render()
    t1 = time.time()
    dt = t1 - t0
    frames_per_sec = num_frames / dt

    # Create an environment with an RGB agent observation
    env = gym.make(env_id, render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    env.reset()
    # Benchmark rendering in agent view
    t0 = time.time()
    for i in range(num_frames):
        obs, reward, terminated, truncated, info = env.step(0)
    t1 = time.time()
    dt = t1 - t0
    agent_view_fps = num_frames / dt

    print(f"Env reset time: {reset_time:.1f} ms")
    print(f"Rendering FPS : {frames_per_sec:.0f}")
    print(f"Agent view FPS: {agent_view_fps:.0f}")

    env.close()


def benchmark_manual_control(env_id, num_resets, num_frames, tile_size):
    env = gym.make(env_id, tile_size=tile_size)
    env = ManualControl(env, seed=args.seed)

    # Benchmark env.reset
    t0 = time.time()
    for i in range(num_resets):
        env.reset()
    t1 = time.time()
    dt = t1 - t0
    reset_time = (1000 * dt) / num_resets

    # Benchmark rendering
    t0 = time.time()
    for i in range(num_frames):
        env.redraw()
    t1 = time.time()
    dt = t1 - t0
    frames_per_sec = num_frames / dt

    # Create an environment with an RGB agent observation
    env = gym.make(env_id, tile_size=tile_size)
    env = RGBImgPartialObsWrapper(env, env.tile_size)
    env = ImgObsWrapper(env)

    env = ManualControl(env, seed=args.seed)
    env.reset()

    # Benchmark rendering in agent view
    t0 = time.time()
    for i in range(num_frames):
        env.step(0)
    t1 = time.time()
    dt = t1 - t0
    agent_view_fps = num_frames / dt

    print(f"Env reset time: {reset_time:.1f} ms")
    print(f"Rendering FPS : {frames_per_sec:.0f}")
    print(f"Agent view FPS: {agent_view_fps:.0f}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        dest="env_id",
        help="gym environment to load",
        default="MiniGrid-LavaGapS7-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--num-resets",
        type=int,
        help="number of times to reset the environment for benchmarking",
        default=200,
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="number of frames to test rendering for",
        default=5000,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )

    args = parser.parse_args()
    benchmark(args.env_id, args.num_resets, args.num_frames)

    benchmark_manual_control(
        args.env_id, args.num_resets, args.num_frames, args.tile_size
    )
