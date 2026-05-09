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
