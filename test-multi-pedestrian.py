import time
import random
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.agents import PedAgent
# from gym_minigrid.envs import MultiPedestrianEnv
# %matplotlib auto
# %load_ext autoreload
# %autoreload 2

# Load the gym environment
env = gym.make('MultiPedestrian-Empty-20x80-v0')
env.addAgents([PedAgent((5, 5), 3, 2), PedAgent((5, 6), 3, 2), PedAgent((5, 4), 3, 2)])
# env = FullyObsWrapper(env)
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# env = ImgObsWrapper(env) # Get rid of the 'mission' field
env.reset()

for i in range(0, 20):
#     print("step {}".format(i))

    obs, reward, done, info = env.step(None)
    # print(obs)
#     print(obs["direction"])
#     print(obs["position"])
    
    if done:
        "Reached the goal"
        break

    env.render()

    time.sleep(0.25)

#     time.sleep(0.05)

# Test the close method
env.close()