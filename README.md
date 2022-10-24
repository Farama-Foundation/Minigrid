<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/minigrid-text.png" width="500px"/>
</p>

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="figures/door-key-curriculum.gif" width=200 alt="Figure Door Key Curriculum">
</p>

The Minigrid library contains a collection of discrete grid-world environments to conduct research on Reinforcement Learning. The environments follow the [Gymnasium]() standard API and they are designed to be lightweight, fast, and easily customizable. 

The documentation website is at [minigrid.farama.org](https://minigrid.farama.org/), and we have a public discord server (which we also use to coordinate development work) that you can join here: [https://discord.gg/B8ZJ92hu](https://discord.gg/B8ZJ92hu)

Note that the library was previously known as gym-minigrid and it has been referenced in several publications. If your publication uses the Minigrid library and you wish for it to be included in the [list of publications](https://minigrid.farama.org/content/publications/), please create an issue in the [GitHub repository](https://github.com/Farama-Foundation/Minigrid/issues/new/choose).


# Installation

To install the Minigrid library use `pip install minigrid`.

We support Python 3.7, 3.8, 3.9 and 3.10 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

# Environments
The included environments can be divided in two groups. The original `Minigrid` environments and the `BabyAI` environments. 

## Minigrid
Following is a list of the environments that were included in the original `Minigrid` library. Each environment provides one or more configurations registered with OpenAI gym. Each environment is also programmatically tunable in terms of size/complexity, which is useful for curriculum learning or to fine-tune difficulty.

* [Empty](https://minigrid.farama.org/environments/empty/) - The goal of the agent is to reach a locatin in an empty room. This environment is useful, with small rooms, to validate that your RL algorithm works correctly, and with large rooms to experiment with sparse rewards and exploration.

* [Four Rooms](https://minigrid.farama.org/environments/four_rooms/) - The agent is randomly placed in a maze with four rooms connected by a small gap. The agent has to reach a randomly placed goal in one of the rooms.

* [Door Key](https://minigrid.farama.org/environments/door_key/) - The agent is placed in a two room maze. The rooms are connected by a closed door and the agent needs to collect a key, open the door, and reach a goal in the contiguous room.

* [Multi Room](https://minigrid.farama.org/environments/multi_room/) - This environment has a series of connected rooms with doors that must be opened in order to get to the next room. The final room has the green goal square the agent must get to. This environment is extremely difficult to solve using RL alone. However, by gradually increasing the number of rooms and building a curriculum, the environment can be solved.

* [Fetch](https://minigrid.farama.org/environments/fetch/) - This environment has multiple objects of assorted types and colors. The agent receives a textual string as part of its observation telling it which object to pick up. Picking up the wrong object terminates the episode with zero reward.

* [Go To Door](https://minigrid.farama.org/environments/go_to_door/) - This environment is a room with four doors, one on each wall. The agent receives a textual (mission) string as input, telling it which door to go to, (eg: "go to the red door"). It receives a positive reward for performing the `done` action next to the correct door, as indicated in the mission string.

* [Go To Object](https://minigrid.farama.org/environments/go_to_object/) - The task to be solved is the same as for the `Go To Door` environments. However, in this case, the mission string asks the agent to reach other type of objects (e.g Box), and perform the `done` action. 

* [Put Near](https://minigrid.farama.org/environments/put_near/) - The agent is instructed through a textual string to pick up an object and place it next to another object.

* [Red Blue Doors](https://minigrid.farama.org/environments/red_blue_doors/) - The agent is randomly placed within a room with one red and one blue door facing opposite directions. The agent has to open the red door and then open the blue door, in that order. Note that, surprisingly, this environment is solvable without memory.

* [Memory](https://minigrid.farama.org/environments/memory/) - This environment is a memory test. The agent starts in a small room where it sees an object. It then has to go through a narrow hallway which ends in a split. At each end of the split there is an object, one of which is the same as the object in the starting room. The agent has to remember the initial object, and go to the matching object at split.

* [Locked Room](https://minigrid.farama.org/environments/memory/) - The environment has six rooms, one of which is locked. The agent receives a textual mission string as input, telling it which room to go to in order to get the key that opens the locked room. It then has to go into the locked room in order to reach the final goal. This environment is extremely difficult to solve with vanilla reinforcement learning alone.

* [Key Corridor](https://minigrid.farama.org/environments/memory/) - This environment is similar to the locked room environment, but there are multiple registered environment configurations of increasing size, making it easier to use curriculum learning to train an agent to solve it. The agent has to pick up an object which is behind a locked door. The key is hidden in another room, and the agent has to explore the environment to find it. The mission string does not give the agent any clues as to where the key is placed. This environment can be solved without relying on language.

* [Unlock](https://minigrid.farama.org/environments/unlock/) - The agent has to open a locked door. This environment can be solved without relying on language.

* [Unlock Pickup](https://minigrid.farama.org/environments/unlock_pickup/) - The agent has to pick up a box which is placed in another room, behind a locked door. This environment can be solved without relying on language.

* [Blocked Unlock Pickup](https://minigrid.farama.org/environments/blocked_unlock_pickup/) - The agent has to pick up a box which is placed in another room, behind a locked door. The door is also blocked by a ball which the agent has to move before it can unlock the door. Hence, the agent has to learn to move the ball, pick up the key, open the door and pick up the object in the other room. This environment can be solved without relying on language.

* [Obstructed Maze](https://minigrid.farama.org/environments/obstructed_maze/) - The agent has to pick up a box which is placed in a corner of a 3x3 maze. The doors are locked, the keys are hidden in boxes and doors are obstructed by balls. This environment can be solved without relying on language.

* [Distributional Shift](https://minigrid.farama.org/environments/dist_shift/) - his environment is based on one of the [DeepMind AI safety gridworlds](https://github.com/deepmind/ai-safety-gridworlds). The agent starts in the top-left corner and must reach the goal which is in the top-right corner, but has to avoid stepping into lava on its way. The aim of this environment is to test an agent's ability to generalize. There are two slightly different variants of the environment, so that the agent can be trained on one variant and tested on the other.

* [Lava Gap](https://minigrid.farama.org/environments/lava_gap/) - The agent has to reach the green goal square at the opposite corner of the room, and must pass through a narrow gap in a vertical strip of deadly lava. Touching the lava terminate the episode with a zero reward. This environment is useful for studying safety and safe exploration.

* [Lava Crossing](https://minigrid.farama.org/environments/simple_crossing/) - The agent has to reach the green goal square on the other corner of the room while avoiding rivers of deadly lava which terminate the episode in failure. Each lava stream runs across the room either horizontally or vertically, and has a single crossing point which can be safely used; Luckily, a path to the goal is guaranteed to exist. This environment is useful for studying safety and safe exploration.

* [Simple Crossing](https://minigrid.farama.org/environments/simple_crossing/) - Similar to the `LavaCrossing` environment, the agent has to reach the green goal square on the other corner of the room, however lava is replaced by walls. This MDP is therefore much easier and maybe useful for quickly testing your algorithms.

* [Dynamic Obstacles](https://minigrid.farama.org/environments/dynamic/) - This environment is an empty room with moving obstacles. The goal of the agent is to reach the green goal square without colliding with any obstacle. A large penalty is subtracted if the agent collides with an obstacle and the episode finishes. This environment is useful to test Dynamic Obstacle Avoidance for mobile robots with Reinforcement Learning in Partial Observability.


## BabyAI
These environments have been imported from the [BabyAI](https://github.com/mila-iqia/babyai) project library. The purpose of this collection of environments is to perform research on grounded language learning. The environments are derived from the `Minigrid` grid-world environments and include an additional functionality that generates synthetic
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
