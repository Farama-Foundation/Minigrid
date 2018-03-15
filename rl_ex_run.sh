#!/usr/bin/env bash
PYTHONPATH=./gym_minigrid/:$PYTHONPATH
PYTHONPATH=./gym_minigrid/envs/:$PYTHONPATH
export PYTHONPATH

python3 pytorch_rl/main.py --env-name MiniGrid-MultiRoom-N6-v0 --no-vis --num-processes 48 --algo a2c