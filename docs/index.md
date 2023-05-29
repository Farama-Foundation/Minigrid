---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/minigrid-text.svg
:alt: Minigrid Logo
```

```{project-heading}
Minigrid contains simple and easily configurable grid world environments to conduct Reinforcement Learning research. This library was previously known as gym-minigrid.
```

```{figure} ../figures/door-key-curriculum.gif
   :alt: door key environment gif
   :width: 350
   :height: 350
```

This library contains a collection of 2D grid-world environments with goal-oriented tasks. The agent in these environments is a triangle-like agent with a discrete action space. The tasks involve solving different maze maps and interacting with different objects such as doors, keys, or boxes.  The design of the library is meant to be simple, fast, and easily customizable.

In addition, the environments found in the [BabyAI](https://github.com/mila-iqia/babyai) repository have been included in Minigrid and will be further maintained under this library.

The Gymnasium interface allows to initialize and interact with the Minigrid default environments as follows:

```{code-block} python

import gymnasium as gym
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```

```{toctree}
:hidden:
:caption: Introduction

content/basic_usage
content/publications
content/create_env_tutorial
```

```{toctree}
:hidden:
:caption: Wrappers

api/wrapper
```


```{toctree}
:hidden:
:caption: Environments

environments/minigrid/index
environments/babyai/index
```

```{toctree}
:hidden:
:caption: Development

release_notes
Github <https://github.com/Farama-Foundation/MiniGrid>
```

