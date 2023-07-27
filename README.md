# PedGrid: A Simple yet Expressive Simulation Environment for Pedestrian Behavior Modeling

PedGrid is a minimal simulator for autonomous vehicle driving with a visualization tool. It's perfect for initial research method development with low computing resources and minimal learning curve. A team of high school and college students is creating challenging reinforcement learning tasks using OpenAI Gymnasium Framework. Currently, we are creating several indoor and outdoor environments for a variety of tasks related to human behavior and motion. Current environments:

1. A corridor environment to model bidirectional pedestrian flow.
2. A two-lane environment to model pedestrian crossing.

Please cite our accepted paper if you use PedGrid or doing relevant research (DOI pending):

```

@inproceedings{inproceedings,
    author = {Muktadir, Golam Md and Huang, Taorui and Ikram, Zarif and Jawad, Abdul and Whitehead, Jim},
    booktitle = {26th IEEE International Conference on Intelligent Transportation Systems ITSC 2023A (Bilbao, Bizkaia, Spain)}
    year = {2023},
    title = {PedGrid - A Simple yet Expressive Simulation Environment for Pedestrian Behavior Modeling}
}
```

# Details and User Guide
We have a seperate website for documentation and tutorials **[here](https://pedgrid.readthedocs.io/)**.

# Why use PedGrid:
1. It's open source
2. Easy to learn (we can get started with research in a week)
3. Easy to setup: written in python using python packages only. 
4. Grid-based: maths are easier to handle.
5. Easy to get metrics: We have a set of commonly used metrics in research. Data is ready for your further analysis.



## Install dependencies via Conda
```
conda config --append channels conda-forge
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