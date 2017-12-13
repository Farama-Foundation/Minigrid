# Minimalistic Grid World Environment (MiniGrid)

Simple and minimailistic grid world environment for OpenAI Gym.

## Installation

Requirements:
- Python 3
- OpenAI gym
- numpy
- PyQT5
- PyTorch (if using the supplied `basicrl` training code)
- matplotlib (if using the supplied `basicrl` training code)

Start by manually installing [PyTorch](http://pytorch.org/).

Then, clone the repository and install the other dependencies with `pip3`:

```
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip3 install -e .
```

## Usage

To run the standalone UI application:

```
./standalone.py
```

The environment being run can be selected with the `--env-name` option, eg:

```
./standalone.py --env-name MiniGrid-Fetch-8x8-v0
```

To see available environments and their implementation, look at [simple_envs.py](gym_minigrid/envs/simple_envs.py).

Basic reinforcement learning code is provided in the `basicrl` subdirectory.
You can perform training using the ACKTR algorithm with:

```
python3 basicrl/main.py --env-name MiniGrid-Empty-8x8-v0 --no-vis --num-processes 32 --algo acktr
```
