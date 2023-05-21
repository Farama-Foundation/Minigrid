import time
import gym
import gym_minigrid
from gym_minigrid.agents import *
import logging
env = gym.make('TwoLaneRoadEnv60x80-v0')       
env.reset()

env.addVehicleAgent(Vehicle(1, (30, 30), (40, 35), 1, 2, 2, 1, 1))

for i in range(110):

    obs, reward, done, info = env.step(None)
    
    if done:
        "Reached the goal"
        break

    env.render()

    if i % 10 == 0:
        logging.info(f"Completed step {i+1}")

    time.sleep(0.5)