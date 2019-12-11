#!/usr/bin/env python3

import time
import argparse
import gym_minigrid
import gym

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name",
    dest="env_name",
    help="gym environment to load",
    default='MiniGrid-DoorKey-8x8-v0'
)
parser.add_argument("--num_frames", default=2000)
args = parser.parse_args()

env = gym.make(args.env_name)
env.reset()

t0 = time.time()

for i in range(args.num_frames):
    env.render('rgb_array')

t1 = time.time()
dt = t1 - t0
dt_per_frame = (1000 * dt) / args.num_frames
frames_per_sec = args.num_frames / dt
print('dt per frame: {:.1f} ms'.format(dt_per_frame))
print('frames_per_sec: {:.1f}'.format(frames_per_sec))
