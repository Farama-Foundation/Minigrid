import time
import random
import gym
import numpy as np
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.agents import PedAgent
# from gym_minigrid.envs import MultiPedestrianEnv
# %matplotlib auto
# %load_ext autoreload
# %autoreload 2

# Load the gym environment
env = gym.make('MultiPedestrian-Empty-20x80-v0')
<<<<<<< Updated upstream
env.addAgents([PedAgent((5, 5), 3, 2), PedAgent((7, 10), 3, 2), PedAgent((2, 5), 3, 2)])
=======
agents = []
for i in range(50):
    xPos = np.random.randint(2, 18)
    yPos = np.random.randint(10, 90)
    direction = 1 if np.random.random() > 0.5 else 3
    prob = np.random.random()
    if prob < 0.1:
        speed = 1
    elif prob < 0.9:
        speed = 2
    else:
        speed = 3
    agents.append(PedAgent((xPos, yPos), direction, speed))
env.addAgents(agents)

>>>>>>> Stashed changes
# env = FullyObsWrapper(env)
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# env = ImgObsWrapper(env) # Get rid of the 'mission' field
env.reset()

for i in range(0, 400):
#     print("step {}".format(i))

    obs, reward, done, info = env.step(None)
    # print(obs)
#     print(obs["direction"])
#     print(obs["position"])
    
    if done:
        "Reached the goal"
        break

    env.render()

    time.sleep(0.05)

#     time.sleep(0.05)

# Test the close method
env.close()
