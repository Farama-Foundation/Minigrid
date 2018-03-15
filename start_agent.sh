#!/usr/bin/env bash
PYTHONPATH=./gym_minigrid/:$PYTHONPATH
PYTHONPATH=./gym_minigrid/envs/:$PYTHONPATH
export PYTHONPATH

envname=$1

if [ $# -eq 0 ]
  then
    echo "Write the name of the environment you want to start"
    echo "MiniGrid-Empty-6x6-v0
MiniGrid-Empty-8x8-v0
MiniGrid-Empty-16x16-v0
MiniGrid-DoorKey-5x5-v0
MiniGrid-DoorKey-6x6-v0
MiniGrid-DoorKey-8x8-v0
MiniGrid-DoorKey-16x16-v0
MiniGrid-MultiRoom-N2-S4-v0
MiniGrid-MultiRoom-N6-v0
MiniGrid-Fetch-5x5-N2-v0
MiniGrid-Fetch-6x6-N2-v0
MiniGrid-Fetch-8x8-N3-v0
MiniGrid-GoToDoor-5x5-v0
MiniGrid-GoToDoor-6x6-v0
MiniGrid-GoToDoor-8x8-v0
MiniGrid-PutNear-6x6-N2-v0
MiniGrid-PutNear-8x8-N3-v0
MiniGrid-LockedRoom-v0
MiniGrid-FourRoomQA-v0
"
fi
fi

python3 pytorch_rl/main.py --env-name $envname --no-vis --num-processes 48 --algo a2c