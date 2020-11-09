#!/usr/bin/env python3

import time
import argparse
import gym_minigrid
import gym
from gym_minigrid.wrappers import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name",
    dest="env_name",
    help="gym environment to load",
    default='MiniGrid-LavaGapS7-v0'
)
parser.add_argument("--num_resets", default=200)
parser.add_argument("--num_frames", default=5000)
args = parser.parse_args()

env = gym.make(args.env_name)

# Benchmark env.reset
t0 = time.time()
for i in range(args.num_resets):
    env.reset()
t1 = time.time()
dt = t1 - t0
reset_time = (1000 * dt) / args.num_resets

# Benchmark rendering
t0 = time.time()
for i in range(args.num_frames):
    env.render('rgb_array')
t1 = time.time()
dt = t1 - t0
frames_per_sec = args.num_frames / dt

# Create an environment with an RGB agent observation
env = gym.make(args.env_name)
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)

# Benchmark rendering
t0 = time.time()
for i in range(args.num_frames):
    obs, reward, done, info = env.step(0)
t1 = time.time()
dt = t1 - t0
agent_view_fps = args.num_frames / dt

print('Env reset time: {:.1f} ms'.format(reset_time))
print('Rendering FPS : {:.0f}'.format(frames_per_sec))
print('Agent view FPS: {:.0f}'.format(agent_view_fps))
