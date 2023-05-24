<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid-text.png" width="500px"/>
</p>

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="figures/door-key-curriculum.gif" width=200 alt="Figure Door Key Curriculum">
</p>

The Minigrid library contains a collection of discrete grid-world environments to conduct research on Reinforcement Learning. The environments follow the [Gymnasium]() standard API and they are designed to be lightweight, fast, and easily customizable. 

The documentation website is at [minigrid.farama.org](https://minigrid.farama.org/), and we have a public discord server (which we also use to coordinate development work) that you can join here: [https://discord.gg/bnJ6kubTg6](https://discord.gg/bnJ6kubTg6)

Note that the library was previously known as gym-minigrid and it has been referenced in several publications. If your publication uses the Minigrid library and you wish for it to be included in the [list of publications](https://minigrid.farama.org/content/publications/), please create an issue in the [GitHub repository](https://github.com/Farama-Foundation/Minigrid/issues/new/choose).


# Installation

To install the Minigrid library use `pip install minigrid`.

We support Python 3.7, 3.8, 3.9 and 3.10 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

# Environments
The included environments can be divided in two groups. The original `Minigrid` environments and the `BabyAI` environments. 

## Minigrid
The list of the environments that were included in the original `Minigrid` library can be found in the [documentation](https://minigrid.farama.org/environments/minigrid/). These environments have in common a triangle-like agent with a discrete action space that has to navigate a 2D map with different obstacles (Walls, Lava, Dynamic obstacles) depending on the environment. The task to be accomplished is described by a `mission` string returned by the observation of the agent. These mission tasks include different goal-oriented and hierarchical missions such as picking up boxes, opening doors with keys or navigating a maze to reach a goal location. Each environment provides one or more configurations registered with Gymansium. Each environment is also programmatically tunable in terms of size/complexity, which is useful for curriculum learning or to fine-tune difficulty.

## BabyAI
These environments have been imported from the [BabyAI](https://github.com/mila-iqia/babyai) project library and the list of environments can also be found in the [documentation](https://minigrid.farama.org/environments/babyai/). The purpose of this collection of environments is to perform research on grounded language learning. The environments are derived from the `Minigrid` grid-world environments and include an additional functionality that generates synthetic
natural-looking instructions (e.g. “put the red ball next to the box on your left”) that command the the agent to navigate the world (including unlocking doors) and move objects to specified locations in order to accomplish the task.

# Training an Agent
The [rl-starter-files](https://github.com/lcswillems/torch-rl) is a repository with examples on how to train `Minigrid` environments with RL algorithms. This code has been tested and is known to work with this environment. The default hyper-parameters are also known to converge. 

# Citation

The original `gym-minigrid` environments were created as part of work done at [Mila](https://mila.quebec). The Dynamic obstacles environment were added as part of work done at [IAS in TU Darmstadt](https://www.ias.informatik.tu-darmstadt.de/) and the University of Genoa for mobile robot navigation with dynamic obstacles.

To cite this project please use:

```
@software{minigrid,
  author = {Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman},
  title = {Minimalistic Gridworld Environment for Gymnasium},
  url = {https://github.com/Farama-Foundation/Minigrid},
  year = {2018},
}
```

If using the `BabyAI` environments please also cite the following:

```
@article{chevalier2018babyai,
  title={Babyai: A platform to study the sample efficiency of grounded language learning},
  author={Chevalier-Boisvert, Maxime and Bahdanau, Dzmitry and Lahlou, Salem and Willems, Lucas and Saharia, Chitwan and Nguyen, Thien Huu and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1810.08272},
  year={2018}
}
```
