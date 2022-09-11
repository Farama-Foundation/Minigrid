---
title: Wrappers
lastpage:
---

# Wrappers

MiniGrid is built to support tasks involving natural language and sparse rewards.
The observations are dictionaries, with an 'image' field, partially observable
view of the environment, a 'mission' field which is a textual string
describing the objective the agent should reach to get a reward, and a 'direction'
field which can be used as an optional compass. Using dictionaries makes it
easy for you to add additional information to observations
if you need to, without having to encode everything into a single tensor.

There are a variety of wrappers to change the observation format available in [minigrid/wrappers.py](../../../../minigrid/wrappers.py). 
If your RL code expects one single tensor for observations, take a look at `FlatObsWrapper`. 
There is also an `ImgObsWrapper` that gets rid of the 'mission' field in observations, leaving only the image field tensor.

Please note that the default observation format is a partially observable view of the environment using a
compact and efficient encoding, with 3 input values per visible grid cell, 7x7x3 values total.
These values are **not pixels**. If you want to obtain an array of RGB pixels as observations instead,
use the `RGBImgPartialObsWrapper`. You can use it as follows:

```python
import gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env) # Get pixel observations
env = ImgObsWrapper(env) # Get rid of the 'mission' field
obs, _ = env.reset() # This now produces an RGB tensor only
```