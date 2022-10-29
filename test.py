import time
import random
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.envs import MultiPedestrianEnv
# %matplotlib auto
# %load_ext autoreload
# %autoreload 2

# Load the gym environment
env = gym.make('MultiPedestrian-Empty-20x80-v0')
env = FullyObsWrapper(env)
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# env = ImgObsWrapper(env) # Get rid of the 'mission' field
env.reset()

for i in range(0, 20):
#     print("step {}".format(i))

    # Pick a random action
#     action = random.randint(0, env.action_space.n - 1)
    actions = []
    for j in range(0, 3):
        actions.append(random.randint(0, 2))

    obs, reward, done, info = env.step(actions)
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