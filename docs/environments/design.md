---
layout: "contents"
title: Design
firstpage:
---

# General Structure

Structure of the world:
- The world is an NxM grid of tiles
- Each tile in the grid world contains zero or one object
  - Cells that do not contain an object have the value `None`
- Each object has an associated discrete color (string)
- Each object has an associated type (string)
  - Provided object types are: wall, floor, lava, door, key, ball, box and goal
- The agent can pick up and carry exactly one object (eg: ball or key)
- To open a locked door, the agent has to be carrying a key matching the door's color

Actions in the basic environment:
- Turn left
- Turn right
- Move forward
- Pick up an object
- Drop the object being carried
- Toggle (open doors, interact with objects)
- Done (task completed, optional)

Default tile/observation encoding:
- Each tile is encoded as a 3 dimensional tuple: `(OBJECT_IDX, COLOR_IDX, STATE)` 
- `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in [minigrid/minigrid.py](minigrid/minigrid.py)
- `STATE` refers to the door state with 0=open, 1=closed and 2=locked

By default, sparse rewards are given for reaching a green goal tile. A
reward of 1 is given for success, and zero for failure. There is also an
environment-specific time step limit for completing the task.
You can define your own reward function by creating a class derived
from `MiniGridEnv`. Extending the environment with new object types or new actions
should be very easy. If you wish to do this, you should take a look at the
[minigrid/minigrid.py](minigrid/minigrid.py) source file.