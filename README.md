# Minimalistic Gridworld Environment for Pedestrian Scenarios (PedGrid)

## Original documentation for learn the basics of MiniGrid code and APIs:
This is a fork of the MiniGrid library. Please, refer to the original documentation library's documentation if you want to use this repository for your research purposes.Requirements:

Python 3.7 to 3.10
OpenAI Gym v0.19 to v0.21
NumPy 1.18+
Matplotlib (optional, only needed for display) - 3.0+

[./documentation/minigrid-basics.md](./documentation/minigrid-basics.md)

## Install dependencies via Conda
```
conda create -n pedgrid python=3.18.13
conda activate pedgrid
conda install gym=0.21
conda install matplotlib
```

## First explore the environmments:

```
python manual_control.py --env Pedestrian-Empty-9x16-v0
python manual_control.py --env MultiPedestrian-Empty-9x16-v0
```