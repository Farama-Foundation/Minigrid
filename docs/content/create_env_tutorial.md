---
layout: "contents"
title: Tutorial on Creating Environments
firstpage:
---

# Tutorial on Creating Environments

In this tutorial, we will go through the process of creating a new environment. 

## Boilerplate Code

```python
class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"
```

First, we need to create a class the inherits from `MiniGridEnv`, we call our class `SimpleEnv`. Then, we define a mission space, the recommended way to do it is to define a static function

```python
@staticmethod
def _gen_mission():
    return "grand mission"
```

that only returns a string which corresponds to the mission. We then pass this function as an argument

```python
mission_space = MissionSpace(mission_func=self._gen_mission)
```

Then, in the `__init__` function, we pass the required arguments to the parent class. In this case we are passing the `mission_space`, `grid_size` and `max_steps`. We also create `self.agent_start_pos` and `self.agent_start_dir` so that member functions can have access to these two values.

## Generate the grid-world

To create your own grid-world environment we override the function `_gen_grid`. We can see from the `MiniGridEnv` class 

```python
# MiniGridEnv._gen_grid
@abstractmethod
def _gen_grid(self, width, height):
    pass
```

`_gen_grid` takes in two inputs `width` and `height`, which are used to specify the size of the environment. 

### Create World

To create the environment, we first an empty grid using

```python
self.grid = Grid(width, height)
```

Then, we create the walls that surrounds the grid

```python
self.grid.wall_rect(0, 0, width, height)
```
Finally, we place the agent in the environment

```python
if self.agent_start_pos is not None:
    self.agent_pos = self.agent_start_pos
    self.agent_dir = self.agent_start_dir
else:
    self.place_agent()
```

these lines of code is saying if we specified the agent starting position and direction the environment will follow what we specified, otherwise, it will randomly place the agent within the environment. If we render the environment right now, it would look like this:

```{figure} ../../figures/tutorial_imgs/first_step.png
:alt: env after first step
:width: 200px
```

### Place Goal

To place a goal in the environment, we use the function

```python
self.put_obj(Goal(), width - 2, height - 2)
```

which places the goal in the bottom right corner. Now the environment should look like this:

```{figure} ../../figures/tutorial_imgs/second_step.png
:alt: env after second step
:width: 200px
```

### Create Separating Walls

To create a wall that separates the environment into two rooms, we use the command

```python
for i in range(0, height):
    self.grid.set(5, i, Wall())
```

this goes over the grids with coordinates `(5, 0)`, `(5, 1)` ... `(5, height)` and make them walls. The result would look like:

```{figure} ../../figures/tutorial_imgs/third_step.png
:alt: env after third step
:width: 200px
```

### Add Keys and Doors

To add keys and doors, we would first need to import

```python
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Door, Key
```

Then, we can simply place the door and key using the command

```python
self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
self.grid.set(3, 6, Key(COLOR_NAMES[0]))
```

Now the environment looks like:

```{figure} ../../figures/tutorial_imgs/fourth_step.png
:alt: env after fourth step
:width: 200px
```

Even for creating more complicated environments, this is all you need know.

## Source Code

The source code of this tutorial is

```python
from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


def main():
    env = SimpleEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()
```