#!/usr/bin/env bash
PYTHONPATH=./gym_minigrid/:$PYTHONPATH
PYTHONPATH=./gym_minigrid/envs/:$PYTHONPATH
export PYTHONPATH

python3 pytorch_rl/enjoy.py --env-name MiniGrid-MultiRoom-N6-v0 --load-dir ./trained_models/a2c