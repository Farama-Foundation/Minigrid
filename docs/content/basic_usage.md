---
layout: "contents"
title: Basic Usage
firstpage:
---


## Basic Usage

There is a UI application which allows you to manually control the agent with the arrow keys:

```
./minigrid/manual_control.py
```

The environment being run can be selected with the `--env` option, eg:

```
./minigrid/manual_control.py --env MiniGrid-Empty-8x8-v0
```

## Training an Agent

If you want to train an agent with reinforcement learning, I recommend using the code found in the [torch-rl](https://github.com/lcswillems/torch-rl) repository. 
This code has been tested and is known to work with this environment. The default hyper-parameters are also known to converge.

A sample training command is:

```
cd torch-rl
python3 -m scripts.train --env MiniGrid-Empty-8x8-v0 --algo ppo
```

