# MiniGrid documentation


This folder contains the documentation for MiniGrid. 

For more information about how to contribute to the documentation go to our [CONTRIBUTING.md](https://github.com/Farama-Foundation/Celshast/blob/main/CONTRIBUTING.md)

### Editing an environment page

If you are editing an Atari environment, directly edit the md file in this repository. 

Otherwise, fork Gym and edit the docstring in the environment's Python file. Then, pip install your Gym fork and run `docs/scripts/gen_mds.py` in this repo. This will automatically generate a md documentation file for the environment.

## Build the Documentation

Install the required packages and Minigrid:

```
pip install -r docs/requirements.txt
pip install -e .
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
