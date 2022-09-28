#!/usr/bin/env python3

import time

import gymnasium as gym

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
    # Benchmark rendering
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        dest="env_id",
        help="gym environment to load",
        default="MiniGrid-LavaGapS7-v0",
    )
    parser.add_argument("--num_resets", default=200)
    parser.add_argument("--num_frames", default=5000)
    args = parser.parse_args()
    benchmark(args.env_id, args.num_resets, args.num_frames)
