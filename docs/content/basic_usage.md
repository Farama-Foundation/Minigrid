---
layout: "contents"
title: Basic Usage
firstpage:
---


## Basic Usage

There is a UI application which allows you to manually control the agent with the arrow keys:

```bash
./minigrid/manual_control.py
```

The environment being run can be selected with the `--env` option, eg:

```bash
./minigrid/manual_control.py --env MiniGrid-Empty-8x8-v0
```

## Installation

Minigrid call be installed via `pip`:

```bash
python3 -m pip install minigrid
```

To modify the codebase or contribute to Minigrid, you would need to install Minigrid from source:

```bash
git clone https://github.com/Farama-Foundation/Minigrid.git
cd Minigrid
python3 -m pip install -e .
```
