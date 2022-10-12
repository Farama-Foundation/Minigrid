---
hide-toc: true
firstpage:
lastpage:
---
## Minigrid (formerly gym-minigrid) contains simple and easily configurable grid world environments for reinforcement learning

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

There are other gridworld Gymnasium environments out there, but this one is
designed to be particularly simple, lightweight and fast. The code has very few
dependencies, making it less likely to break or fail to install. It loads no
external sprites/textures, and it can run at up to 5000 FPS on a Core i7
laptop, which means you can run your experiments faster. A known-working RL
implementation can be found [in this repository](https://github.com/lcswillems/torch-rl).

Requirements:
- Python 3.7 to 3.10
- Gymnasium v0.26
- NumPy 1.18+
- Matplotlib (optional, only needed for display) - 3.0+

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{minigrid,
  author = {Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman},
  title = {Minimalistic Gridworld Environment for Gymnasium},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Farama-Foundation/MiniGrid}},
}
```


## Installation

There is now a [pip package](https://pypi.org/project/minigrid/) available, which is updated periodically:

```
pip install minigrid
```

Alternatively, to get the latest version of MiniGrid, you can clone this repository and install the dependencies with `pip3`:

```
git clone https://github.com/Farama-Foundation/MiniGrid
cd MiniGrid
pip install -e .
```

```{toctree}
:hidden:
:caption: Introduction

content/installation
content/basic_usage
content/publications
```

```{toctree}
:hidden:
:caption: Wrappers

api/wrapper
```


```{toctree}
:hidden:
:caption: Environments

environments/design
environments/index
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/MiniGrid>
Donate <https://farama.org/donations>
Contribute to the Docs <https://github.com/Farama-Foundation/MiniGrid/blob/master/.github/PULL_REQUEST_TEMPLATE.md>
```

