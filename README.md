# Minimalistic Grid World Environment (MiniGrid)

There are other grid world Gym environments out there, but this one is
designed to be especially simple, lightweight and fast. The code has very few
dependencies, loads no external sprites/textures, and can run at up to 5800
FPS on my quad-core i7 laptop.

## Installation

Clone this repository and install the other dependencies with `pip3`:

```
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip3 install -e .
```

Optionally, if you wish use the reinforcement learning code included
under [/basicrl](/basicrl), you can install its dependencies as follows:

```
cd basicrl

# PyTorch
conda install pytorch torchvision -c soumith

# OpenAI baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

Note: the basicrl code is a custom fork of [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
which was modified to work with this environment.

## Usage

To run the standalone UI application, which allows you to manually control the agent with the arrow keys:

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

You can view the result of training using the `enjoy.py` script:

```
python3 basicrl/enjoy.py --env-name MiniGrid-Empty-8x8-v0 --load-dir ./trained_models/acktr
```
