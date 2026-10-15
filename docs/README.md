# Minigrid documentation

This folder contains the documentation for Minigrid.

For more information about how to contribute to the documentation go to our [CONTRIBUTING.md](https://github.com/Farama-Foundation/Celshast/blob/main/CONTRIBUTING.md)


## Build the Documentation

Install the required packages and Minigrid:

```
pip install -r docs/requirements.txt
pip install -e .
```

Generate the documentation files for the environments:

```
python docs/_scripts/gen_env_docs.py
python docs/_scripts/gen_envs_display.py
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml . _build
```

## Regenerate the DoorKey animation

The DoorKey animation shows a tabular Q-learning policy trained with
`FullyObsWrapper` on `MiniGrid-DoorKey-5x5-v0`. The policy learns from the
environment's sparse reward using full-grid observations. Training uses seed 73
and 10,000 episodes; the recorded episode uses seed 123.

From the repository root, run:

```
python docs/_scripts/gen_doorkey_gif.py
```

This takes about two minutes on a CPU and needs NumPy and Pillow in addition to
Minigrid. It replaces `docs/_static/videos/minigrid/DoorKeyEnv.gif` only after a
successful rollout. `gen_gifs.py` preserves this animation when regenerating
the random-policy animations for other environments.
